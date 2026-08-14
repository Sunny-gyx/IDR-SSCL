import bisect
import logging
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from torchvision.models import resnet50

from src.models.cbm import CBM_SSL
import src.train.utils as utils


def _posterior_dominance_mask(values, gmm, mean_rank):
    """Select a mean-ranked GMM component by strict posterior dominance."""
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    component_order = np.argsort(gmm.means_.reshape(-1))
    target_component = component_order[mean_rank]
    probabilities = gmm.predict_proba(values)
    competitors = np.delete(probabilities, target_component, axis=1)
    return np.all(
        probabilities[:, target_component, None] > competitors,
        axis=1,
    )


class Ours_CEM(CBM_SSL):
    """IDR-SSCL with concept embedding model (CEM) representations."""
    def __init__(
            self,
            n_concepts,
            n_tasks,
            emb_size=16,
            training_intervention_prob=0.25,
            embedding_activation="leakyrelu",
            shared_prob_gen=True,
            concept_loss_weight=1,
            concept_loss_weight_labeled=1,
            concept_loss_weight_unlabeled=5,
            task_loss_weight=1,

            c2y_model=None,
            c2y_layers=None,
            c_extractor_arch=utils.wrap_pretrained_model(resnet50),
            output_latent=False,

            optimizer="adam",
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            weight_loss=None,
            task_class_weights=None,
            k=5,

            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_policy=None,
            output_interventions=False,
            use_concept_groups=False,

            top_k_accuracy=None,
            pos_weight=None,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(active_intervention_values)
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(inactive_intervention_values)
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            if embedding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(self.pre_concept_model.modules())[-1].out_features,
                            2 * emb_size,
                        ),
                    ])
                )
            elif embedding_activation == "sigmoid":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(self.pre_concept_model.modules())[-1].out_features,
                            2 * emb_size,
                        ),
                        torch.nn.Sigmoid(),
                    ])
                )
            elif embedding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(self.pre_concept_model.modules())[-1].out_features,
                            2 * emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embedding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(self.pre_concept_model.modules())[-1].out_features,
                            2 * emb_size,
                        ),
                        torch.nn.ReLU(),
                    ])
                )
            if self.shared_prob_gen and (
                    len(self.concept_prob_generators) == 0
            ):
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
        if c2y_model is None:
            units = [
                        n_concepts * emb_size
                    ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()

        self.register_buffer('cpt_pos_weight', pos_weight)
        self.loss_concept = torch.nn.BCELoss()
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.use_concept_groups = use_concept_groups
        self.k=k

        self.eval_int=False
        self.lesion_grading_matrix = torch.zeros((n_tasks,n_concepts*self.emb_size)).to(self.device)

    def _after_interventions(
            self,
            prob,
            pos_embeddings,
            neg_embeddings,
            intervention_idxs=None,
            c_true=None,
            train=False,
            competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
                (c_true is not None) and
                (intervention_idxs is None)
        ):
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true, intervention_idxs

    def _forward(
            self,
            x,
            c=None,
            y=None,
            l=None,
            train=False,
            latent=None,
            intervention_idxs=None,
            competencies=None,
            prev_interventions=None,
            output_embeddings=False,
            output_latent=None,
            output_interventions=None
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []
            logits=[]

            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                if prob.dim() == 1:
                    prob = prob.unsqueeze(-1)  # [B, 1]
                contexts.append(torch.unsqueeze(context, dim=1))
                logits.append(prob)
                c_sem.append(self.sig(prob))
            logits = torch.cat(logits, dim=-1)
            c_sem = torch.cat(c_sem, dim=-1)
            contexts = torch.cat(contexts, dim=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        if not train and c is not None and l is not None and self.eval_int:
            intervene_sample_idx = l.nonzero().squeeze()
            
            if intervene_sample_idx.numel() > 0:
                c_sem_labeled = c_sem[l]
                
                lesion_lbls = c[l]
                
                concept_logits_labeled = logits[l]

                int_precision = torch.clamp(lesion_lbls, 0.01, 0.99)
                
                target_scale = (
                    torch.log(int_precision / (1 - int_precision)) / concept_logits_labeled
                )

                lesion_int_scale = torch.where(
                    lesion_lbls == c_sem_labeled.round(), torch.tensor(1.0, device=x.device), target_scale
                )
                
                mask = torch.zeros_like(c_sem_labeled)
                mask[:, :] = 1
                
                concept_logits_i_labeled = concept_logits_labeled * (1 - mask) + concept_logits_labeled * mask * lesion_int_scale
                
                c_sem_intervened_labeled = self.sig(concept_logits_i_labeled)
                
                c_sem_updated = c_sem.clone()
                c_sem_updated[l] = c_sem_intervened_labeled
                
                c_sem = c_sem_updated
                
                intervention_idxs = intervene_sample_idx.to(x.device)
                c_int = c
            else:
                c_int = c
        else:
            c_int = c
            intervention_idxs = None

        c_pred = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(c_sem, dim=-1) +
                contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(c_sem, dim=-1))
        )
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        tail_results = []
        if output_interventions:
            if (intervention_idxs is not None) and isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(intervention_idxs).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])
        return tuple([logits,c_sem, c_pred, y] + tail_results)
    
    def _run_step(
            self,
            batch,
            batch_idx,
            train=False,
            intervention_idxs=None,
    ):
        if self.current_epoch<20:
            x, y, c, l,nbr_c,nbr_w= self._unpack_batch(batch)

            outputs = self._forward(
                x,
                c=c,
                y=y,
                l=l,
                train=train,
                competencies=None,
                prev_interventions=None,
                intervention_idxs=intervention_idxs,
            )
            c_logits,c_sem,c_pred, y_logits = outputs[0], outputs[1], outputs[2],outputs[3]

            if self.task_loss_weight != 0:
                task_loss = self.loss_task(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)
                task_loss_scalar = task_loss.detach()
            else:
                task_loss = 0
                task_loss_scalar = 0

            if self.concept_loss_weight != 0:
                if(c[l].numel()==0):
                    concept_loss=torch.tensor(0.0,device=c[l].device,requires_grad=True)
                else:
                    concept_loss = self.loss_concept(c_sem[l], c[l])

                concept_loss_scalar = (concept_loss).detach()
                loss = self.concept_loss_weight * (concept_loss) + task_loss
            else:
                loss = task_loss
                concept_loss_scalar = 0.0
        else:
            x, y, c, l,nbr_c,nbr_w= self._unpack_batch(batch)
            pseudo_label=batch['pseudo_label']
            pseudo=batch['pseudo']

            outputs = self._forward(
                x,
                c=c,
                y=y,
                l=l,
                train=train,
                competencies=None,
                prev_interventions=None,
                intervention_idxs=intervention_idxs,
            )
            c_logits,c_sem,c_pred, y_logits = outputs[0], outputs[1], outputs[2],outputs[3]

            if self.task_loss_weight != 0:
                task_loss = self.loss_task(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)
                task_loss_scalar = task_loss.detach()
            else:
                task_loss = 0
                task_loss_scalar = 0

            if self.concept_loss_weight != 0:
                if(torch.cat([c[l],pseudo_label[pseudo]]).numel()==0):
                    concept_loss=torch.tensor(0.0,device=c[l].device,requires_grad=True)
                else:
                    concept_loss = self.loss_concept(torch.cat([c_sem[l],c_sem[pseudo]]),  torch.cat([c[l],pseudo_label[pseudo]]))

                concept_loss_scalar = (concept_loss).detach()
                loss = self.concept_loss_weight * (concept_loss) + task_loss
            else:
                loss = task_loss
                concept_loss_scalar = 0.0
            
        result={}
        result["loss"]=loss
        result["c_loss_labeled"]=concept_loss_scalar
        result["task_loss"]=task_loss_scalar
        result["c_sem"]=c_logits
        result["c"]=c
        result["y"]=y
        result["y_pred"]=y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1)
        return loss,result 
    
    def on_train_epoch_start(self):
        if self.current_epoch % 5==0 and self.current_epoch>=20:
            anchor = self._anchor_ext()
            unlabel = self._anchor_sim()
            
            self.ld_modeling_update()

            lpul = PLUL(
            anchor, unlabel,self.device,f"pseudo_selection/Ours_CEM/image_{self.current_epoch}.png",self.k
            )
            
            origin_l_sel_idxs, origin_l_sel_p = lpul.get_new_label()
            logging.info("DASS pseudo candidates: %d", len(origin_l_sel_idxs))
            l_sel_idxs,l_sel_p=self.pseudo_selection(origin_l_sel_idxs,origin_l_sel_p)
            logging.info(f"Final Pseudo selection: {len(l_sel_idxs)}")

            anchor_idxs = lpul.anchor_purify()
            origin_a_sel_idxs, origin_a_sel_p = lpul.get_new_anchor(anchor_idxs)
            logging.info("DASS anchor candidates: %d", len(origin_a_sel_idxs))
            a_sel_idxs,a_sel_p=self.pseudo_selection(origin_a_sel_idxs,origin_a_sel_p)
            logging.info(f"Final Anchor selection: {len(a_sel_idxs)}")

            for i in range(len(l_sel_idxs)):
                dataset_idx=bisect.bisect_right(self.trainer.train_dataloader.dataset.cumulative_sizes, l_sel_idxs[i])
                if dataset_idx==0:
                    sample_idx=l_sel_idxs[i]
                else:
                    sample_idx=l_sel_idxs[i]-self.trainer.train_dataloader.dataset.cumulative_sizes[dataset_idx-1]
                self.trainer.train_dataloader.dataset.datasets[dataset_idx].update(sample_idx,l_sel_p[i],"pseudo")

            for i in range(len(a_sel_idxs)):
                dataset_idx=bisect.bisect_right(self.trainer.train_dataloader.dataset.cumulative_sizes, a_sel_idxs[i])
                if dataset_idx==0:
                    sample_idx=a_sel_idxs[i]
                else:
                    sample_idx=a_sel_idxs[i]-self.trainer.train_dataloader.dataset.cumulative_sizes[dataset_idx-1]
                self.trainer.train_dataloader.dataset.datasets[dataset_idx].update(sample_idx,a_sel_p[i],"anchor")

    def _anchor_ext(self):
        p1, logits1, embed1, gts, idxs = (
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
        )
        with torch.no_grad():
            for batch in self.trainer.train_dataloader:
                inputs=batch["img0"]
                labels=batch["lesion_label"]
                disease_lbl=batch['drgrading_level']
                item=batch["idx"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                anchor=batch["anchor"]
                l=batch["l"]
                anchor_label=batch["anchor_label"]

                
                if torch.cat([inputs[l],inputs[anchor]]).numel() > 0:
                    outputs = self._forward(
                       x=torch.cat([inputs[l],inputs[anchor]]),
                        c=torch.cat([labels[l],labels[anchor]]),
                        y=torch.cat([disease_lbl[l],disease_lbl[anchor]]),
                        l=l,
                        train=False,
                        competencies=None,
                        prev_interventions=None,
                        intervention_idxs=None,
                    )
                    c_logits,c_sem,c_pred, y_logits = outputs[0], outputs[1], outputs[2],outputs[3]
                    feat1=F.normalize(c_pred, dim=-1,p=2)           
                    item = torch.from_numpy(item.numpy()).to(self.device)
                    embed1 = torch.cat((embed1, feat1))
                    idxs = torch.cat((idxs, torch.cat([item[l],item[anchor]])))
                    gts = torch.cat((gts, torch.cat([labels[l],anchor_label[anchor].to(self.device)])))
                    logits1 = torch.cat((logits1, c_logits))
                    p1 = torch.cat((p1, c_sem))
            return edict(
                {
                    "embed1": embed1,
                    "idxs": idxs,
                    "gts": gts,
                    "p1": p1,
                    "logits1": logits1,
                }
            )
    
    def _anchor_sim(self):
        u_gts,u_idxs,u_preds1,u_embed1,u_logits1 = (
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
        )
        with torch.no_grad():
            for batch in self.trainer.train_dataloader:
                inputs=batch["img0"]
                labels=batch["lesion_label"]
                disease_lbl=batch['drgrading_level']
                item=batch["idx"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                l=batch["l"]
                pseudo=batch["pseudo"]

                if inputs[~(l) & ~(pseudo)].numel() > 0:
                    outputs = self._forward(
                       x=inputs[~(l) & ~(pseudo)],
                        c=labels[~(l) & ~(pseudo)],
                        y=disease_lbl[~(l) & ~(pseudo)],
                        l=l,
                        train=False,
                        competencies=None,
                        prev_interventions=None,
                        intervention_idxs=None,
                    )
                    c_logits,c_sem,c_pred, y_logits = outputs[0], outputs[1], outputs[2],outputs[3]
                    feat1=F.normalize(c_pred, dim=-1,p=2)           
                    item = torch.from_numpy(item.numpy()).to(self.device)
                    u_embed1 = torch.cat((u_embed1, feat1))
                    u_idxs = torch.cat((u_idxs, item[~(l) & ~(pseudo)]))
                    u_gts = torch.cat((u_gts, labels[~(l) & ~(pseudo)]))
                    u_logits1 = torch.cat((u_logits1, c_logits))
                    u_preds1 = torch.cat((u_preds1, c_sem))
            return edict(
                {
                    "embed1": u_embed1,
                    "idxs": u_idxs,
                    "gts": u_gts,
                    "p1": u_preds1,
                    "logits1": u_logits1,
                }
            )
    
    def ld_modeling_update(self):
        """Build one concept-representation prototype per DR grade for LGPS."""
        p=[[]for _ in range(self.n_tasks)]
        for batch in self.trainer.train_dataloader:
            inputs = batch["img0"].to(self.device)
            lesion_labels = batch["lesion_label"].to(self.device)
            pseudo = batch["pseudo"]
            l = batch["l"]
            disease_lbls=batch['drgrading_level'].to(self.device)

            with torch.no_grad():
                if torch.cat([inputs[l], inputs[pseudo]]).numel() > 0:
                    outputs = self._forward(
                        x=torch.cat([inputs[l], inputs[pseudo]]),
                        c=lesion_labels,
                        y=None, l=l, train=False,
                        competencies=None, prev_interventions=None, intervention_idxs=None,
                    )
                    c_logits, c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2], outputs[3]
                    c_pred_l=c_pred[:l.sum()]
                    if isinstance(disease_lbls, torch.Tensor) and disease_lbls.device != torch.device('cpu'):
                        disease_lbls_cpu = disease_lbls.cpu()
                    else:
                        disease_lbls_cpu = disease_lbls

         
            for sample_idx, grading in enumerate(disease_lbls_cpu[l].tolist()):
                p[grading].append(c_pred_l[sample_idx:sample_idx+1].detach().to(self.device))
        
        for i in range(self.n_tasks):
            if len(p[i]) > 0:
                stacked = torch.cat(p[i], dim=0)          # [Ni, D]
                proto = stacked.mean(dim=0, keepdim=True) # [1, D]
            else:
                proto = torch.zeros((1, self.emb_size*self.n_concepts), device=self.device)
            self.lesion_grading_matrix[i, :] = proto.squeeze(0)

    def pseudo_selection(self, origin_l_sel_idxs, origin_l_sel_p):
        """Apply CPLV followed by LGPS to DASS-selected candidates."""
        dataset = self.trainer.train_dataloader.dataset
    
        imgs  = []
        imgs_s = []
        for idx in origin_l_sel_idxs:
            data_dict = dataset[int(idx)]
            imgs.append(data_dict["img"])
            imgs_s.append(data_dict["img_s"])

        imgs  = torch.stack(imgs).to(self.device)      # shape: (N, C, H, W)
        imgs_s = torch.stack(imgs_s).to(self.device)   # shape: (N, C, H, W)

        with torch.no_grad():
            _, pred1, c_pred1, y_logits1 = self._forward(x=imgs,   c=None, y=None, l=None, train=False,
                                        competencies=None, prev_interventions=None, intervention_idxs=None)
            _, pred2, _, _ = self._forward(x=imgs_s, c=None, y=None, l=None, train=False,
                                        competencies=None, prev_interventions=None, intervention_idxs=None)

        loss = F.binary_cross_entropy(pred2, pred1, reduction='none')   # (N, n_tasks)
        per_sample = loss.mean(dim=1).cpu()  # (N,)

        discrepancies = per_sample.numpy().reshape(-1, 1).astype(np.float64)
        discrepancy_gmm = GaussianMixture(
            n_components=2,
            max_iter=100,
            tol=1e-3,
            reg_covar=5e-4,
            random_state=42,
        ).fit(discrepancies)
        trust_mask = _posterior_dominance_mask(
            discrepancies,
            discrepancy_gmm,
            mean_rank=0,
        )
        filtered_l_sel_idxs = origin_l_sel_idxs[trust_mask]
        filtered_l_sel_p = origin_l_sel_p[trust_mask]
        prototype=c_pred1[trust_mask]
        predicted_grades=torch.argmax(y_logits1, dim=1).cpu().numpy()[trust_mask]

        logging.info(f"Pseudo selection after consistency filtering: {len(filtered_l_sel_idxs)}")
        similarity=[]
        for i in range(len(filtered_l_sel_idxs)):
            grading=predicted_grades[i]
            sim=F.cosine_similarity(prototype[i,:],self.lesion_grading_matrix[grading,:].to(self.device),dim=0)
            similarity.append(sim.item())
        similarity=np.asarray(similarity, dtype=np.float64).reshape(-1, 1)
        similarity_gmm = GaussianMixture(
            n_components=2,
            max_iter=100,
            tol=1e-3,
            reg_covar=5e-4,
            random_state=42,
        ).fit(similarity)
        sim_mask = _posterior_dominance_mask(
            similarity,
            similarity_gmm,
            mean_rank=-1,
        )
        l_sel_idxs=filtered_l_sel_idxs[sim_mask]
        l_sel_p=filtered_l_sel_p[sim_mask]
        return l_sel_idxs, l_sel_p

class PLUL:
    """Distribution-aware sample selection (DASS) over local density."""
    def __init__(
        self, x_info, u_info,device,log_path,k,ds_mixup=False, num_gmm_sets=3
    ) -> None:
        self.x_info = x_info
        self.u_info = u_info
        self.ds_mixup = ds_mixup
        self.device = device
        self.log_path = log_path

        self.local_info = self.build_local_graph(
            x_info["embed1"].cpu().numpy(),
            u_info["embed1"].cpu().numpy(),
            k,
        )

        self.local_ds = self.get_ds()
        self.idxs_pack = self.build_GMM(num_gmm_sets=num_gmm_sets, fig=False)

        self.sel = self.idxs_pack[2]
        self.local_agg = self.get_agg()

        self.local_info_r = self.build_local_graph(
            u_info["embed1"].cpu().numpy(), x_info["embed1"].cpu().numpy(), k
        )

    def anchor_purify(self):
        result_idxs = []
        for i in range(self.local_info["i"][self.sel].shape[0]):
            unlabel_node = self.local_info["i"][self.sel[i]]
            mask = np.asarray(
                [
                    self.sel[i] in self.local_info_r["i"][unlabel_node[j]]
                    for j in range(unlabel_node.shape[0])
                ]
            )

            mask_idxs = mask.nonzero()[0]
            result_idxs.append(mask_idxs)

        result_mask = np.asarray([i.shape[0] for i in result_idxs])
        anchor_idxs = self.sel[(result_mask == result_mask.min()).nonzero()[0]]
        return anchor_idxs

    def build_local_graph(self, x_embed, u_embed, k):
        logging.info("Building local graph with k=%d", k)
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        nbrs.fit(x_embed)
        distances, indices = nbrs.kneighbors(u_embed)
        similarities = 1 - distances
        return edict({"d": similarities, "i": indices})

    def get_ds(self):
        ds = self.local_info["d"].mean(1)
        ds = (ds - ds.min()) / (ds.max() - ds.min())
        ds = ds.reshape(-1, 1)
        return ds

    def get_agg(self):
        knn_gts = self.x_info["gts"][self.local_info["i"]].cpu().numpy()
        agg = knn_gts.mean(1)
        return edict({"knn_gts": knn_gts, "agg": agg})

    def get_pseudo(self, target):
        pred = self.u_info["p1"][target].cpu().numpy()
        agg = self.local_agg["agg"][target]
        if self.ds_mixup:
            weight = self.local_ds[target]
        else:
            weight = 1.0
        return weight * pred + (1 - weight) * agg

    def get_new_label(self):
        sel_idxs = self.u_info["idxs"][self.sel].cpu().numpy().astype(int)
        sel_pseudo = self.get_pseudo(self.sel)
        return sel_idxs, sel_pseudo

    def get_new_anchor(self, sel):
        sel_idxs = self.u_info["idxs"][sel].cpu().numpy().astype(int)
        sel_pseudo = self.get_pseudo(sel)
        return sel_idxs, sel_pseudo

    def build_GMM(self, num_gmm_sets=3, fig=False, name="Local Density"):
        """Fit DASS GMM and group samples by mean-ranked posterior dominance."""
        target = self.local_ds
        gmm1 = GaussianMixture(
            n_components=num_gmm_sets,
            max_iter=20,
            tol=1e-2,
            reg_covar=5e-7,
            random_state=1,
        )

        gmm1.fit(target)
        high_saliency_target = np.flatnonzero(
            _posterior_dominance_mask(target, gmm1, mean_rank=0)
        )
        low_saliency_target = np.flatnonzero(
            _posterior_dominance_mask(target, gmm1, mean_rank=-1)
        )

        if num_gmm_sets == 3:
            medium_saliency_target = np.flatnonzero(
                _posterior_dominance_mask(target, gmm1, mean_rank=1)
            )

        if fig:
            plt.hist(
                target[low_saliency_target],
                bins=200,
                range=(0.0, 1.0),
                edgecolor="black",
                alpha=0.5,
                label=f"High {name}",
            )
            plt.hist(
                target[high_saliency_target],
                bins=200,
                range=(0.0, 1.0),
                edgecolor="black",
                alpha=0.5,
                label=f"Informative {name}",
            )
            if num_gmm_sets == 3:
                plt.hist(
                    target[medium_saliency_target],
                    bins=200,
                    range=(0.0, 1.0),
                    edgecolor="black",
                    alpha=0.5,
                    label=f"Uncertain {name}",
                )
            plt.legend()
            plt.grid()
            plt.savefig(self.log_path)
            plt.clf()
        if num_gmm_sets == 3:
            return low_saliency_target, medium_saliency_target, high_saliency_target
        else:
            return low_saliency_target, high_saliency_target

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
from tqdm import tqdm

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


class Ours_CBM(CBM_SSL):
    """IDR-SSCL with a conventional concept bottleneck model (CBM)."""
    def __init__(
            self,
            n_concepts,
            n_tasks,
            concept_loss_weight=1,
            concept_loss_weight_labeled=1,
            concept_loss_weight_unlabeled=5,
            task_loss_weight=1,

            extra_dims=0,
            bool=False,
            sigmoidal_prob=True,
            sigmoidal_extra_capacity=True,
            bottleneck_nonlinear=None,
            output_latent=False,

            x2c_model=None,
            c_extractor_arch=utils.wrap_pretrained_model(resnet50),
            c2y_model=None,
            c2y_layers=None,

            optimizer="adam",
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            weight_loss=None,
            task_class_weights=None,
            k=5,
            dropout_dl=None,

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
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.output_interventions = output_interventions
        if x2c_model is not None:
            self.x2c_model = x2c_model
        else:
            self.x2c_model = c_extractor_arch(output_dim=(n_concepts + extra_dims))

        if c2y_model is not None:
            self.c2y_model = c2y_model
        else:
            units = [n_concepts + extra_dims] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.FloatTensor(active_intervention_values)
        else:
            self.active_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]) * (5.0 if not sigmoidal_prob else 1.0)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.FloatTensor(inactive_intervention_values)
        else:
            self.inactive_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]) * (-5.0 if not sigmoidal_prob else 0.0)

        self.sigmoid = torch.nn.Sigmoid()
        if sigmoidal_extra_capacity:
            bottleneck_nonlinear = "sigmoid"
        if bottleneck_nonlinear == "sigmoid":
            self.bottleneck_nonlin = torch.nn.Sigmoid()
        elif bottleneck_nonlinear == "leakyrelu":
            self.bottleneck_nonlin = torch.nn.LeakyReLU()
        elif bottleneck_nonlinear == "relu":
            self.bottleneck_nonlin = torch.nn.ReLU()
        elif (bottleneck_nonlinear is None) or (
                bottleneck_nonlinear == "identity"
        ):
            self.bottleneck_nonlin = lambda x: x
        else:
            raise ValueError(
                f"Unsupported nonlinearity '{bottleneck_nonlinear}'"
            )
        
        self.register_buffer('cpt_pos_weight', pos_weight)
        self.loss_concept = torch.nn.BCELoss()
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
            )
        )
        self.bool = bool
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_extra_capacity = sigmoidal_extra_capacity
        self.use_concept_groups = use_concept_groups

        self.k=k
        self.eval_int=False
        self.lesion_grading_matrix = torch.zeros((n_tasks,n_concepts+extra_dims)).to(self.device)
    
    def _standardize_indices(self, intervention_idxs, batch_size):
        if isinstance(intervention_idxs, list):
            intervention_idxs = np.array(intervention_idxs)
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.IntTensor(intervention_idxs)

        if intervention_idxs is None or (
                isinstance(intervention_idxs, torch.Tensor) and
                ((len(intervention_idxs) == 0) or intervention_idxs.shape[-1] == 0)
        ):
            return None
        if not isinstance(intervention_idxs, torch.Tensor):
            raise ValueError(
                f'Unsupported intervention indices {intervention_idxs}'
            )
        if len(intervention_idxs.shape) == 1:
            intervention_idxs = torch.tile(
                torch.unsqueeze(intervention_idxs, 0),
                (batch_size, 1),
            )
        elif len(intervention_idxs.shape) == 2:
            assert intervention_idxs.shape[0] == batch_size, (
                f'Expected intervention indices to have batch size {batch_size} '
                f'but got intervention indices with shape {intervention_idxs.shape}.'
            )
        else:
            raise ValueError(
                f'Intervention indices should have 1 or 2 dimensions. Instead '
                f'we got indices with shape {intervention_idxs.shape}.'
            )
        if intervention_idxs.shape[-1] == self.n_concepts:
            elems = torch.unique(intervention_idxs)
            if len(elems) == 1:
                is_binary = (0 in elems) or (1 in elems)
            elif len(elems) == 2:
                is_binary = (0 in elems) and (1 in elems)
            else:
                is_binary = False
        else:
            is_binary = False
        if not is_binary:
            intervention_idxs = intervention_idxs.to(dtype=torch.long)
            result = torch.zeros(
                (batch_size, self.n_concepts),
                dtype=torch.bool,
                device=intervention_idxs.device,
            )
            result[:, intervention_idxs] = 1
            intervention_idxs = result
        assert intervention_idxs.shape[-1] == self.n_concepts, (
            f'Unsupported intervention indices with shape {intervention_idxs.shape}.'
        )
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs

    def _concept_intervention(
            self,
            c_pred,
            intervention_idxs=None,
            c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return c_pred
        c_pred_copy = c_pred.clone()
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=c_pred.shape[0],
        )
        intervention_idxs = intervention_idxs.to(c_pred.device)
        if self.extra_dims:
            set_intervention_idxs = torch.nn.functional.pad(
                intervention_idxs,
                pad=(0, self.extra_dims),  # Just pads the last dimension
            )
        else:
            set_intervention_idxs = intervention_idxs
        if self.sigmoidal_prob:
            c_pred_copy[set_intervention_idxs] = c_true[intervention_idxs]
        else:
            active_intervention_values = self.active_intervention_values.to(c_pred.device)
            batched_active_intervention_values = torch.tile(
                torch.unsqueeze(active_intervention_values, 0),
                (c_pred.shape[0], 1)).to(c_true.device)
            inactive_intervention_values = self.inactive_intervention_values.to(c_pred.device)
            batched_inactive_intervention_values = torch.tile(
                torch.unsqueeze(inactive_intervention_values, 0),
                (c_pred.shape[0], 1)).to(c_true.device)
            c_pred_copy[set_intervention_idxs] = (
                    (c_true[intervention_idxs] * batched_active_intervention_values[intervention_idxs]) + (
                    (c_true[intervention_idxs] - 1) * -batched_inactive_intervention_values[intervention_idxs])
            )
        return c_pred_copy

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
            latent = self.x2c_model(x)
        
        if not train and c is not None and l is not None and self.eval_int:
            intervene_sample_idx = l.nonzero().squeeze()
            if intervene_sample_idx.numel() > 0:
                
                concept_logits_labeled = latent[l]
                lesion_lbls = c[l]
                
                c_sem_labeled = self.sigmoid(concept_logits_labeled)

                int_precision = torch.clamp(lesion_lbls, 0.01, 0.99)
                target_scale = (torch.log(int_precision / (1 - int_precision)) / concept_logits_labeled)
                lesion_int_scale = torch.where(
                    lesion_lbls == c_sem_labeled.round(), torch.tensor(1.0, device=x.device), target_scale
                )
                mask = torch.zeros_like(c_sem_labeled)
                mask[:, :] = 1
                
                concept_logits_i_labeled = concept_logits_labeled * (1 - mask) + concept_logits_labeled * mask * lesion_int_scale

                latent_updated = latent.clone()
                latent_updated[l] = concept_logits_i_labeled
                latent = latent_updated
                

        if self.sigmoidal_prob or self.bool:
            if self.extra_dims:
                c_pred_probs = self.sigmoid(latent[:, :-self.extra_dims])
                c_others = self.bottleneck_nonlin(latent[:, -self.extra_dims:])
                c_pred = torch.cat([c_pred_probs, c_others], dim=-1)
                c_sem = c_pred_probs
            else:
                c_pred = self.sigmoid(latent)
                c_sem = c_pred
        else:
            c_pred = latent
            if self.extra_dims:
                c_sem = self.sigmoid(latent[:, :-self.extra_dims])
            else:
                c_sem = self.sigmoid(latent)
        pos_embeddings = torch.ones(c_sem.shape).to(x.device)
        neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
        if output_embeddings or (intervention_idxs is None) and (c is not None) and (
                self.intervention_policy is not None) and not (self.sigmoidal_prob or self.bool):
            if (self.active_intervention_values is not None) and (self.inactive_intervention_values is not None):
                active_intervention_values = self.active_intervention_values.to(c_pred.device)
                pos_embeddings = torch.tile(active_intervention_values, (c.shape[0], 1)
                                            ).to(active_intervention_values.device)
                inactive_intervention_values = self.inactive_intervention_values.to(c_pred.device)
                neg_embeddings = torch.tile(inactive_intervention_values, (c.shape[0], 1)
                                            ).to(inactive_intervention_values.device)
            else:
                out_embs = c_pred.detach().cpu().numpy()
                for concept_idx in range(self.n_concepts):
                    pos_embeddings[:, concept_idx] = np.percentile(out_embs[:, concept_idx], 95)
                    neg_embeddings[:, concept_idx] = np.percentile(out_embs[:, concept_idx], 5)
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)

        if (intervention_idxs is None) and (c is not None) and (self.intervention_policy is not None):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=None
            )
        else:
            c_int = c
        if self.bool:
            y = self.c2y_model((c_pred > 0.5).float())
        else:
            y = self.c2y_model(c_pred)

        tail_results = []
        if output_interventions:
            if intervention_idxs is None:
                intervention_idxs = None
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)
        return tuple([latent,c_sem, c_pred, y] + tail_results)
    
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
            else:
                concept_loss=torch.tensor(0.0,device=c[l].device,requires_grad=True)
                concept_loss_scalar = 0.0

            loss=self.concept_loss_weight * (concept_loss) + task_loss
            
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
            else:
                concept_loss=torch.tensor(0.0,device=c[l].device,requires_grad=True)
                concept_loss_scalar = 0.0
            
            loss=self.concept_loss_weight * (concept_loss) + task_loss
            
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
            anchor, unlabel,self.device,f"pseudo_selection/Ours_CBM/image_{self.current_epoch}.png",self.k
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
            for batch in tqdm(self.trainer.train_dataloader,desc="anchor_ext"):
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
                    feat1=F.normalize(c_logits, dim=-1,p=2)           
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
            for batch in tqdm(self.trainer.train_dataloader,desc="anchor_sim"):
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
                    feat1=F.normalize(c_logits, dim=-1,p=2)           
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
        """Build one concept-logit prototype per DR grade for LGPS."""
        p=[[]for _ in range(self.n_tasks)]
        for batch in tqdm(self.trainer.train_dataloader,desc="Loss Distribution Modeling"):
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
                    c_logits_l=c_logits[:l.sum()]
                    if isinstance(disease_lbls, torch.Tensor) and disease_lbls.device != torch.device('cpu'):
                        disease_lbls_cpu = disease_lbls.cpu()
                    else:
                        disease_lbls_cpu = disease_lbls
            
            for sample_idx, grading in enumerate(disease_lbls_cpu[l].tolist()):
                p[grading].append(c_logits_l[sample_idx:sample_idx+1].detach().to(self.device))
        
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
            c_logits1, pred1, _, y_logits1 = self._forward(x=imgs,   c=None, y=None, l=None, train=False,
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
        prototype=c_logits1[trust_mask]
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

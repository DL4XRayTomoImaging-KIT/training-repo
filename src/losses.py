from segmenetation_losses_pytorch import *
from torch.nn.functional import normalize
from torch import nn
import math


class ContrastiveLoss(nn.Module):
    def __init__(self, device, n_negatives, n_tasks, background_share=0.1, temperature=0.07):
        """
        Contrastive loss for Semantic Segmetation implementation, based on https://arxiv.org/pdf/2102.06191.pdf
        :param device: torch.device, device of the data and model
        :param n_negatives: int, number of negatives found for each example
        :param n_tasks: int, number of classes for segmentation (includes both organs and tumors)
        :param temperature: float, temperature value for the exponent
        """
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives
        # not used in the current version
        self.n_background = math.ceil(n_negatives * background_share)
        self.n_task = n_negatives - self.n_background

        self.task_list = torch.arange(n_tasks * 2).unsqueeze(1)  # .to(device)
        self.one = torch.Tensor([1.]).to(device)
        self.device = device

    def forward(self, features, labels, tasks):
        """
        Calclulate contrastive loss for d-dimentional representations of clusters
        :param features: torch.FloatTensor[b_sz, d, h, w], segmentation feature map
        :param labels: torch.FloatTensor[b_sz, 2, h, w], labels mask 0th channel -- organ, 1st channel -- tumor
        :param tasks: torch.LongTensor[b_sz], tasks corresponding to each image,
                                              each task is next mapped to 2 * task, 2 * task + 1,
                                              even values for organs, odd for tumors
        :return: torch.FloatTensor[1], loss value
        """
        batch_size = features.size(0)

        features, labels = features.unsqueeze(1), labels.unsqueeze(2)
        # is label non empty
        label_mask = labels.sum(dim=(-1, -2)).view(-1, batch_size * 2).type(torch.bool).cpu()

        repr_targets = self.prepare_reprs(features, labels)
        repr_backgrounds = self.prepare_reprs(features, 1 - labels)

        negatives, positives = self.gather_negatives_and_positives(tasks, label_mask,
                                                                   repr_targets, repr_backgrounds)

        # only calculate loss for non empty masks
        filter_ids = torch.arange(2 * batch_size)[label_mask.squeeze(0)]
        features = repr_targets[filter_ids].unsqueeze(-1)

        pos_logits = torch.matmul(positives, features).squeeze(-1)
        neg_logits = torch.matmul(negatives, features).squeeze(-1)

        loss = self.logits_to_loss(pos_logits, neg_logits)
        return loss

    def prepare_reprs(self, features, labels):
        """
        Calculate normalized d-dimentional representations for organs, tumors or backgrounds
        :param features: torch.FloatTensor[b_sz, d, h, w], segmentation feature map
        :param labels: torch.FloatTensor[b_sz, 2, h, w], labels mask 0th channel -- organ, 1st channel -- tumor,
                can be inverted for background representation
        :return: torch.FloatTensor[b_sz * 2, d] mean representation of the organs and tumors or their backgrounds
        """
        masked = features * labels
        repr_vectors = masked.sum(dim=(-1, -2))
        # if labels sum is 0, mean is also 0
        cnt = torch.maximum(labels.sum(dim=(-1, -2)), self.one)
        repr_vectors /= cnt

        d = repr_vectors.size(-1)
        repr_vectors = repr_vectors.view(-1, d)
        repr_vectors = normalize(repr_vectors)
        return repr_vectors

    def gather_negatives_and_positives(self, tasks, label_mask, repr_targets, repr_backgrounds):
        """
        Collect one positive and self.n_negatives examples for each item in the batch
        :param tasks: torch.LongTensor[b_sz], tasks corresponding to each image
        :param label_mask: torch.BoolTensor[1, 2 * b_sz], binary mask of whether
                the image contains something relevant to the task
        :param repr_targets: torch.FloatTensor[b_sz * 2, d], mean representation of the organs and tumors
        :param repr_backgrounds: torch.FloatTensor[b_sz * 2, d], mean representation of the backgrounds
        :return: tuple(torch.FloatTensor[b_sz, self.n_negatives, d], torch.FloatTensor[b_sz, 1, d])
               negative examples (of different task or background of the same task),
               positive examples (same task, but not same item)
        """
        batch_size = len(tasks)

        task_ids = torch.zeros((1, batch_size * 2), dtype=torch.long)
        task_ids[:, ::2] = 2 * tasks  # organs
        task_ids[:, 1::2] = 2 * tasks + 1  # tumors

        loss_calc_idx = task_ids[label_mask].squeeze(0)

        # [self.n_tasks, batch_size * 2] for each task, all items that are not of the same task and non-empty
        cooc_other = (self.task_list != task_ids) * label_mask

        # [self.n_tasks, batch_size * 2] for each task, all items that are of the same task
        cooc_same = (self.task_list == task_ids)

        target_idx_neg = cooc_other[loss_calc_idx]
        background_idx_neg = cooc_same[loss_calc_idx]

        self_exclude_mask = torch.ones_like(target_idx_neg).scatter_(1, loss_calc_idx.unsqueeze(1), 0)
        target_idx_pos = cooc_same[loss_calc_idx] * self_exclude_mask

        target_negatives = [repr_targets[target_index] for target_index in target_idx_neg]
        background_negatives = [repr_backgrounds[background_index] for background_index in background_idx_neg]

        positives = torch.stack(tuple(repr_targets[target_index][:1] for target_index in target_idx_pos))

        negatives = tuple(map(self.sample_negatives, zip(target_negatives, background_negatives)))
        negatives = torch.stack(negatives)

        return negatives, positives

    def logits_to_loss(self, pos_logits, neg_logits):
        """
        Convert logits to contrastive loss value: -log (exp(x_pos / temp) / sum(exp(x_neg / temp)))
        :param pos_logits: torch.FloatTensor[b_sz, self.n_negatives, d]
        :param neg_logits: torch.FloatTensor[b_sz, 1, d]
        :return: torch.FloatTensor[1] the loss value
        """
        pos_logits /= self.temperature
        neg_logits /= self.temperature

        neg_logits_max = torch.max(neg_logits, dim=1, keepdim=True)[0].detach()
        neg_logits -= neg_logits_max
        pos_logits -= neg_logits_max
        loss = (torch.log(torch.exp(neg_logits).sum(dim=1)) - pos_logits.squeeze(1)).mean()
        return loss

    def sample_negatives(self, neg):
        """
        Select self.n_negatives negative examples
        :param neg: Tuple(torch.FloatTensor[1, n_target_negatives], torch.FloatTensor[1, n_background_negatives])
                    Note: n_target_negatives and n_background_negatives are not fixed!
        :return: torch.FloatTensor[1, self.n_negatives]
        """
        target_negatives, background_negatives = neg
        all_negatives = torch.cat((target_negatives, background_negatives))
        idx = torch.randperm(all_negatives.size(1))[: self.n_negatives]
        negatives = all_negatives[idx]
        return negatives

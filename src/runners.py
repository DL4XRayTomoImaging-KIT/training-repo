from catalyst import dl
import torch
import torch.nn.functional as F


class SelfSuperRunner(dl.Runner):
    def predict_batch(self, batch):
        embeddings = self.model.module.get_embeddings(batch)
        head_preds = self.model.module.predict_heads(embeddings)
        return head_preds

    def _handle_batch(self, batch):
        img, labels, tasks = batch
        features = self.model(img)
        loss = self.criterion(features, labels, tasks)
        self.batch_metrics.update(
            {"loss": loss}
        )


class EmbeddingRunner(dl.Runner):
    def _handle_batch(self, batch):
        emb, labels, tasks = batch
        d = emb.size(-1)
        emb = emb.view(-1, d, 1, 1)
        features = self.model.module.train_head(emb, tasks[0]).view(-1, 1)
        labels = labels.view(-1, 1)
        loss = self.criterion(features, labels)

        self.batch_metrics.update({"loss": loss})
        probs = F.sigmoid(features)
        self.output = {"probs": probs, "preds": probs > 0.5}
        self.input = {"probs": probs, "targets": labels}


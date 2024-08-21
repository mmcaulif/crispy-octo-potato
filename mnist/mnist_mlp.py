import pytorch_lightning as L
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

# https://huggingface.co/docs/datasets/quickstart#vision
from datasets import load_dataset

device = th.device("cuda" if th.cuda.is_available() else "cpu")

ds = load_dataset('mnist', split='train').with_format("torch", device=device)
_transforms = Compose([ToTensor()])
def transforms(examples):
    examples["image"] = [_transforms(img) for img in examples["image"]]
    return examples

ds = ds.with_transform(transforms)

def collate_fn(examples):
    images = []
    labels = []

    for example in examples:
        images.append((example["image"]))
        labels.append(example["label"])        

    pixel_values = th.stack(images)
    labels = th.tensor(labels)
    return {'images': pixel_values, 'labels': labels}


class MnistClassifier(L.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch['images'], batch['labels']
        x = x.view(x.size(0), -1)
        y_hat = self.classifier(x).float()
        y_oh = F.one_hot(y, num_classes=10).float()
        loss = F.binary_cross_entropy(y_hat, y_oh)
        return loss
    
    def predict_step(self, x):
        return self.classifier(x.view(x.size(0), -1)).float()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(-1),
        )

    def forward(self, x):
        return self.net(x)
    
# model
model = MnistClassifier(MLP())

# train model
dataloader = DataLoader(ds, collate_fn=collate_fn, batch_size=1024, num_workers=11)
trainer = L.Trainer(max_epochs=20, enable_checkpointing=False)
trainer.fit(model=model, train_dataloaders=dataloader)

train_acc = 0
for batch in dataloader:    
    x, y = batch['images'], batch['labels']
    y_hat = model.predict_step(x).float()
    y_oh = F.one_hot(y, num_classes=10).float()
    batch_train_acc = th.sum(y_hat == y_oh)
    train_acc += batch_train_acc

print(train_acc/len(ds))
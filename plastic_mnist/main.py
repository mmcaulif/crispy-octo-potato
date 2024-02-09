
import numpy as np
from dataset import MnistDataset
import numpy as np
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState 
import flax.linen as nn
import optax


class ShuffledMnistDataset(MnistDataset):
    def shuffle(self):
        print("shuffling")
        np.random.shuffle(self.train_labels)


class MlpClassifier(nn.Module):
    @nn.compact
    def __call__(self, x):
        z = nn.Dense(2048)(x)
        z = nn.relu(z)
        z = nn.Dense(2048)(z)
        z = nn.relu(z)
        z = nn.Dense(2048)(z)
        z = nn.relu(z)
        z = nn.Dense(1024)(z)
        z = nn.relu(z)
        z = nn.Dense(512)(z)
        z = nn.relu(z)
        z = nn.Dense(10)(z)
        return z


def main():
    # train_data = MnistDataset(True)
    train_data = ShuffledMnistDataset(True)
    train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=True)

    model = MlpClassifier()
    batch = jnp.ones((1, 784))  # (N, H, W, C) format
    params = model.init(jax.random.key(0), batch)
    # output = model.apply(params, batch)

    optim = optax.adam(1e-3)
    opt_state = optim.init(params)

    train_state = TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=optim,
        opt_state=opt_state
    )

    @jax.value_and_grad
    def loss_fn(params, x, y):
        predictions = train_state.apply_fn(params, x)
        # print(predictions)
        loss = optax.softmax_cross_entropy(logits=predictions, labels=y).mean()
        # print(loss)
        # exit()
        return loss

    for epoch in range(1_000):
        avg_loss = []
        avg_acc = []
        
        for data in train_dataloader:

            train_images, train_labels = data
            train_images = train_images.numpy()
            train_labels = train_labels.numpy()

            loss, grads = loss_fn(
                train_state.params,
                train_images, 
                train_labels)
            
            predictions = jnp.exp(nn.log_softmax(train_state.apply_fn(train_state.params, train_images)))
            # print(predictions)
            predictions = jnp.eye(10)[jnp.argmax(predictions, axis=-1)]
            # print(predictions)
            # exit()
            accuracy = (train_labels * predictions).sum() / len(train_labels)

            avg_loss.append(loss)
            avg_acc.append(accuracy)
            # jax.debug.print(str(loss))
            train_state = train_state.apply_gradients(grads=grads)

        avg_loss = np.mean(np.array(avg_loss))
        avg_acc = np.mean(np.array(avg_acc))
        print(f"Epoch {epoch}, steps: {train_state.step}, loss: {avg_loss}, accuray: {avg_acc}")

        if epoch % 1_000 == 0:
            train_dataloader.dataset.shuffle()


if __name__ == '__main__':
    main()
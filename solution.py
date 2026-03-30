import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data(path='data.csv'):
    X, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row['x1']), float(row['x2'])])
            y.append(int(row['label']))
    return np.array(X), np.array(y, dtype=float)


def load_weights(path='model_weights.npz'):
    d = np.load(path)
    return d['W1'], d['b1'], d['W2'], d['b2']


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def forward(X, weights):
    W1, b1, W2, b2 = weights
    b1 = b1.reshape(-1, 1)
    b2 = b2.reshape(-1, 1)

    z1 = W1 @ X.T + b1
    A1 = relu(z1)
    z2 = W2 @ A1+ b2
    A2 = sigmoid(z2)
    return A2, None

def bce_loss(y_hat, y):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    res = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    loss = -np.sum(res)/y.shape[0]

    return loss

def compute_gradients(X, y, weights):
    W1, b1, W2, b2 = weights
    n =y.shape[0]
    b1 = b1.reshape(-1, 1)
    b2 = b2.reshape(-1, 1)

    z1  = W1 @ X.T + b1
    A1 = relu(z1)
    z2 = W2 @ A1 + b2

    dz2 = sigmoid(z2) - y.reshape(1, -1)
    dw2 = dz2 @ A1.T
    db2 = np.sum(dz2, axis=1, keepdims=True)

    dA1 = W2.T @ dz2
    dz1 = dA1 * relu_grad(z1)
    dw1 = dz1 @ X
    db1 = np.sum(dz1, axis=1, keepdims=True)
    return dw1/n, db1/n, dw2/n, db2/n


def gradient_check(X, y, weights, eps=1e-5):
    dw1, db1, dw2, db2 = compute_gradients(X, y, weights)
    grads = [dw1, db1, dw2, db2]
    names = ['W1', 'b1', 'W2', 'b2']
    res = {}
    pairs = list(zip(weights, grads, names))

    for k in range(len(pairs)):
        W, grad_true, name = pairs[k]
        shape = W.shape
        W = W.flatten()
        grad_true = grad_true.flatten()

        grad_num = np.zeros_like(W)
        for i in range(len(W)):
            W_plus = W.copy()
            W_minus = W.copy()

            W_plus[i] += eps
            W_minus[i] -= eps

            w_plus = [w.copy() for w in weights]
            w_minus = [w.copy() for w in weights]

            w_plus[k] = W_plus.reshape(shape)
            w_minus[k] = W_minus.reshape(shape)

            l2 = input_gradient(X, y, w_plus)
            l1 = input_gradient(X, y, w_minus)

            grad_num[i] = (l2 - l1) / (2 * eps)

        abs_diff = np.max(np.abs(grad_num - grad_true))
        rel_diff = np.max(
            np.abs(grad_num - grad_true) /
            (np.abs(grad_num) + np.abs(grad_true) + 1e-15)
        )

        res[name] = {
            'passed': rel_diff < 1e-4,
            'max_abs_diff': abs_diff,
            'max_rel_diff': rel_diff
        }

    return res


def input_gradient(x, y_true, weights):
    y_hat, _ = forward(x, weights)
    return bce_loss(y_hat, y_true)

def grad_z2(x, weights):
    W1, b1, W2, b2 = weights
    x = x.reshape(-1, 1)
    z1 = W1 @ x + b1.reshape(-1, 1)
    a1 = relu(z1)
    z2 = (W2 @ a1 + b2.reshape(-1, 1)).item()
    da1 = W2.T * 1
    dz1 = da1 * relu_grad(z1)
    grad = (W1.T @ dz1).reshape(-1)

    return grad, z2

def pgd_attack(X, y, weights, lr=0.05, steps=200):
    n_samples = X.shape[0]
    deltas = np.zeros_like(X)
    success = np.zeros(n_samples, dtype=bool)

    y_hat, _ = forward(X, weights)
    pred = (y_hat > 0.5).astype(int).flatten()
    correct_mask = (pred == y)

    attack_indices = np.where(correct_mask)[0]
    for idx in attack_indices:
        x_orig = X[idx].copy()
        x_adv = x_orig.copy()
        y_true = y[idx]
        target = 1 - y_true
        delta = np.zeros_like(x_orig)

        for step in range(steps):
            grad, z2 = grad_z2(x_adv, weights)
            pred_current = 1 if z2 > 0 else 0

            if pred_current == target:
                deltas[idx] = delta
                success[idx] = True
                break

            if y_true == 1:
                delta -= lr * grad
            else:
                delta += lr * grad

            x_adv = x_orig + delta

    return deltas, success, correct_mask


def plot_decision_boundary(X, y, weights, deltas, success, correct_mask,
                           save_path='adversarial_plot.png'):
    x_min, x_max = X[:,0].min()-0.3, X[:,0].max()+0.3
    y_min, y_max = X[:,1].min()-0.3, X[:,1].max()+0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz, _ = forward(grid, weights)
    zz = zz.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, title, show_adv in zip(axes,
            ['Start + decision boundary',
             'Adversarial examples'],
            [False, True]):

        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1],
                    colors=['#aec6e8','#f4a97a'], alpha=0.35)
        ax.contour(xx, yy, zz, levels=[0.5],
                   colors='#333333', linewidths=1.2)

        ax.scatter(X[y==0, 0], X[y==0, 1], c='#3578b5', s=14,
                   alpha=0.6, label='Class 0', zorder=3)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='#d6604d', s=14,
                   alpha=0.6, label='Class 1', zorder=3)

        if show_adv:
            adv_idx = np.where(success)[0]
            X_adv = X[adv_idx] + deltas[adv_idx]
            norms = np.linalg.norm(deltas[adv_idx], axis=1)

            sc = ax.scatter(X_adv[:, 0], X_adv[:, 1],
                            c=norms, cmap='YlOrRd',
                            s=30, zorder=5, edgecolors='k',
                            linewidths=0.3, label='Adversarial')
            plt.colorbar(sc, ax=ax, label='‖δ‖₂')

            for j in adv_idx[:60]:
                ax.annotate('', xy=X[j]+deltas[j], xytext=X[j],
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            lw=0.5, alpha=0.5))
            ax.set_title(f'{title}\n'
                         f'{success.sum()} successfull attacks '
                         f'{correct_mask.sum()} correct prerdictions\n'
                         f'Median value ‖δ‖₂ = {np.median(norms):.3f}')
        else:
            ax.set_title(title)

        ax.legend(fontsize=8, markerscale=1.5)
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")



if __name__ == '__main__':
    print("=" * 60)
    print("Loading weights")
    print("=" * 60)
    X, y = load_data('data.csv')
    weights = load_weights('model_weights.npz')
    W1, b1, W2, b2 = weights
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"W1: {W1.shape}, b1: {b1.shape}, W2: {W2.shape}, b2: {b2.shape}")

    print("\n" + "=" * 60)
    print("Verifying forward pass")
    print("=" * 60)
    y_hat, _ = forward(X, weights)
    ref = np.load('reference_predictions.npy')
    max_diff = np.abs(y_hat - ref).max()
    print(f"Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, "ERROR: bad forward pass!"
    print("Forward pass is ok (< 1e-5)")

    acc = ((y_hat > 0.5) == y.astype(bool)).mean()
    print(f"Dataset accuracy: {acc:.4f}")

    print("\n" + "=" * 60)
    print("Gradient check")
    print("=" * 60)
    idx = np.random.choice(len(X), 50, replace=False)
    gc_results = gradient_check(X[idx], y[idx], list(weights))

    all_passed = True
    for name, res in gc_results.items():
        status = 'ok' if res['passed'] else 'error'
        print(f"  {status} {name:3s}  max_abs_diff={res['max_abs_diff']:.2e}"
              f"  max_rel_diff={res['max_rel_diff']:.2e}"
              f"  {'PASS' if res['passed'] else 'FAIL'}")
        if not res['passed']:
            all_passed = False
    print("Gradients verified" if all_passed
          else "Error in gradients!")

    print("\n" + "=" * 60)
    print("Adversarial examples")
    print("=" * 60)
    deltas, success, correct_mask = pgd_attack(X, y, weights,
                                                lr=0.05, steps=300)

    norms = np.linalg.norm(deltas[success], axis=1)
    print(f"Correct predictions: {correct_mask.sum()}")
    print(f"Successfull attacks: {success.sum()}")
    print(f"‖δ‖₂ — min: {norms.min():.4f}")
    print(f"‖δ‖₂ — median: {np.median(norms):.4f}")
    print(f"‖δ‖₂ — max: {norms.max():.4f}")

    plot_decision_boundary(X, y, weights, deltas, success, correct_mask)

from __future__ import annotations
import time, numpy as np, paddle, matplotlib.pyplot as plt

device = 'cuda' if paddle.cuda.is_available() else 'cpu'

def train(model, save_path, train_loader, val_loader, epochs=10, lr=0.0001):
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=lr)
    loss_fn = paddle.nn.CrossEntropyLoss()
    losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_accs = 0
    since = time.time()
    for epoch in range(epochs):
        model = model.to(device).train()
        print(f'Epoch {epoch+1}/{epochs}')
        batch_losses, batch_accs = [], []
        for batch_idx, batch in enumerate(train_loader):
            img, label = batch
            img, label = img.to(device), label.to(device)
            logits = model(img)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(logits, label)
            batch_losses.append(loss.item())
            train_acc = paddle.metric.accuracy(logits, label.reshape([-1, 1]))
            batch_accs.append(train_acc.item())
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_losses[-1]:.4f}, Train Acc: {batch_accs[-1]*100:.2f}%')
        losses.append(np.mean(batch_losses))
        train_accs.append(np.mean(batch_accs))
        print(f'Loss: {losses[-1]:.4f}, Train Acc: {train_accs[-1]*100:.2f}%')
        model = model.to(device).eval()
        with paddle.no_grad():
            batch_losses, batch_accs = [], []
            current_accs = 0
            for batch_idx, batch in enumerate(val_loader):
                img, label = batch
                img, label = img.to(device), label.to(device)
                logits = model(img)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred = paddle.argmax(logits, axis=1)
                loss = loss_fn(logits, label)
                batch_losses.append(loss.item())
                val_acc = (pred == label.flatten()).astype('float32').mean()
                batch_accs.append(val_acc.item())
                current_accs += val_acc
            if best_accs < current_accs / len(val_loader):
                best_accs = current_accs / len(val_loader)
                paddle.save(model.state_dict(), save_path)
            val_losses.append(np.mean(batch_losses))
            val_accs.append(np.mean(batch_accs))
        print(f'Val Loss: {np.mean(val_losses):.4f}, Val Acc: {np.mean(val_accs)*100:.2f}%')
        print('-'*32)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    return losses, val_losses, train_accs, val_accs

def test(model, test_loader):
    model.to(device).eval()
    loss_fn = paddle.nn.CrossEntropyLoss()
    test_losses, test_accs = [], []
    with paddle.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img, label = batch
            img, label = img.to(device), label.to(device)
            logits = model(img)
            pred = paddle.argmax(logits, axis=1)
            loss = loss_fn(logits, label)
            test_losses.append(loss.item())
            test_acc = (pred == label.flatten()).astype('float32').mean()
            test_accs.append(test_acc.item())
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx+1}/{len(test_loader)} Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]*100:.2f}%')
    print(f'Test Loss: {np.mean(test_losses):.4f}, Test Acc: {np.mean(test_accs)*100:.2f}%')

def figure(losses, val_losses, train_accs, val_accs):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses, 'ro-', label='Train Loss')
    plt.plot(val_losses, 'bo-', label='Valid Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'rs-',label='Train Acc')
    plt.plot(val_accs, 'bs-', label='Valid Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()
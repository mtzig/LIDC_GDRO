def train(dataloader, steps_per_epoch, model, loss_fn, optimizer):
    
    model.train()
    for i in range(steps_per_epoch):

        loss = loss_fn(next(dataloader))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, steps_per_epoch, batch_size, model, loss_fn, is_gdro):
    num_batches = steps_per_epoch

    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in range(epoch_size):
            minibatch = next(dataloader)
            if is_gdro:
                X = torch.cat([m[0] for m in minibatch])
                y = torch.cat([m[1] for m in minibatch])
            else:
                X, y = minibatch

            pred = model(X)

            test_loss += loss_fn(minibatch).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= num_batches * batch_size * (3 * is_gdro + 1)

    print("Average Loss:", test_loss, "\nAccuracy:", correct)

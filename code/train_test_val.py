from train_test_step import *

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        loss = train_step(model, data, optimizer, criterion)
        total_loss += loss
    average_loss = total_loss / len(train_loader)
    return average_loss

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            loss, _ = test_step(model, data, criterion)
            val_loss += loss
    average_val_loss = val_loss / len(val_loader)
    return average_val_loss

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            loss, output = test_step(model, data, criterion)
            test_loss += loss
            predictions.append(output)
    average_test_loss = test_loss / len(test_loader)
    return average_test_loss, predictions

def train_epochs(model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler, epochs):
    trainLossPlt = []
    valLossPlt = []
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}')
        trainLossPlt.append(train_loss)

        val_loss = validate(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}')
        valLossPlt.append(val_loss)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

    test_loss, predictions = test(model, test_loader, criterion)
    print(f'Test Loss: {test_loss}')

    return trainLossPlt, valLossPlt, test_loss, predictions


def train_CV(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        loss = train_step(model, data, optimizer, criterion)
        total_loss += loss
    average_loss = total_loss / len(train_loader)
    return average_loss

def validate_CV(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            loss, _ = test_step(model, data, criterion)
            val_loss += loss
    average_val_loss = val_loss / len(val_loader)
    return average_val_loss

def train_epochs_CV(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs):
    trainLossPlt = []
    valLossPlt = []
    for epoch in range(epochs):
        train_loss = train_CV(model, train_loader, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}')
        trainLossPlt.append(train_loss)

        val_loss = validate(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}')
        valLossPlt.append(val_loss)

        # Update the learning rate
        scheduler.step(val_loss)

    return trainLossPlt, valLossPlt
from mxnet import autograd, gluon, image, nd
from mxnet.gluon import nn, data as gdata, loss as gloss, utils as gutils
from FCDensenet import *
from data import *
import time
import mxnet as mx

dropout_rate=0.2
num_channels=48
growth_rate=16
num_classes=11
numconvs_in_downpath=[4,5,7,10,12,15]
numconvs_in_uppath=[12,10,7,5,4]
batch_size=4
input_shape=(384,480)
def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    if isinstance(data_iter, mx.io.MXDataIter):
        data_iter.reset()
    for batch in data_iter:
        features, labels, batch_size = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1)==y).sum().copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar() / n
def _get_batch(batch, ctx):
    """return features and labels on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        features = batch.data[0]
        labels = batch.label[0]
    else:
        features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])
def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train and evaluate a model."""
    print("training on", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_iter, mx.io.MXDataIter):
            train_iter.reset()
        start = time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in ys])
            if print_batches and (i+1) % print_batches == 0:
                print("batch %d, loss %f, train acc %f" % (
                    n, train_l_sum / n, train_acc_sum / m
                ))
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" % (
            epoch, train_l_sum / n, train_acc_sum / m, test_acc, time() - start
        ))
def train_FCDensenet():
    net = FCDensenet(num_channels,dropout_rate,growth_rate,num_classes,numconvs_in_downpath,numconvs_in_uppath)
    net.initialize()
    #x = nd.zeros((batch_size, 3, input_shape[0],input_shape[1]))
    #print net(x).shape
    ctx = mx.gpu(0)
    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1,
                                                     'wd':1e-3})
    voc_train = VOCSegDataset(True, input_shape)
    voc_test = VOCSegDataset(False, input_shape)
    train_data = gluon.data.DataLoader(
        voc_train, batch_size, shuffle=True, last_batch='discard')
    test_data = gluon.data.DataLoader(
        voc_test, batch_size, last_batch='discard')
    train(train_data, test_data, net, loss, trainer, ctx, num_epochs=10)

train_FCDensenet()
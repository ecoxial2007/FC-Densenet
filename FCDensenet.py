from mxnet.gluon import nn
from mxnet import nd

def conv_block(num_channels,dropout_rate):
    blk=nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels,kernel_size=3, padding=1),
            nn.Dropout(dropout_rate)
            )
    return blk
def transdown_block(num_channels,dropout_rate):
    tdb=nn.HybridSequential()
    tdb.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels,kernel_size=1),
            nn.Dropout(dropout_rate),
            nn.MaxPool2D(pool_size=2,strides=2)
            )
    return tdb
def transup_block(num_channels):
    tub=nn.HybridSequential()
    tub.add(nn.Conv2DTranspose(num_channels,kernel_size=4,padding=1,strides=2))
    return tub
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, dropout_rate, **kwargs):
        super(DenseBlock, self).__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels,dropout_rate))

    def forward(self, x):
        for blk in self.net:
            y=blk(x)
            x=nd.concat(x,y,dim=1)
        return x
class FCDensenet(nn.Block):
    def __init__(self,num_channels,dropout_rate,growth_rate,num_classes,numconvs_in_downpath,numconvs_in_uppath,**kwargs):
        super(FCDensenet,self).__init__()
        self.P1 = nn.Sequential()
        self.P2 = nn.Sequential()
        self.P3 = nn.Sequential()
        self.P4 = nn.Sequential()
        self.P5 = nn.Sequential()
        self.P6 = nn.Sequential()

        self.U5 = nn.Sequential()
        self.U4 = nn.Sequential()
        self.U3 = nn.Sequential()
        self.U2 = nn.Sequential()
        self.U1 = nn.Sequential()
        self.final = nn.Sequential()

        self.P1.add(nn.Conv2D(channels=num_channels,kernel_size=3,padding=1),
                    nn.BatchNorm(),
                    nn.Activation('relu'))
        self.P1.add(DenseBlock(numconvs_in_downpath[0],growth_rate,dropout_rate))
        num_channels+=numconvs_in_downpath[0]*growth_rate
        self.P2.add(transdown_block(num_channels,dropout_rate))

        self.P2.add(DenseBlock(numconvs_in_downpath[1], growth_rate, dropout_rate))
        num_channels += numconvs_in_downpath[1] * growth_rate
        self.P3.add(transdown_block(num_channels, dropout_rate))

        self.P3.add(DenseBlock(numconvs_in_downpath[2], growth_rate, dropout_rate))
        num_channels += numconvs_in_downpath[2] * growth_rate
        self.P4.add(transdown_block(num_channels, dropout_rate))

        self.P4.add(DenseBlock(numconvs_in_downpath[3],growth_rate,dropout_rate))
        num_channels+=numconvs_in_downpath[3]*growth_rate
        self.P5.add(transdown_block(num_channels,dropout_rate))

        self.P5.add(DenseBlock(numconvs_in_downpath[4],growth_rate,dropout_rate))
        num_channels+=numconvs_in_downpath[4]*growth_rate
        self.P6.add(transdown_block(num_channels,dropout_rate))

        self.P6.add(DenseBlock(numconvs_in_downpath[5], growth_rate, dropout_rate))

        num_channels = numconvs_in_downpath[5] * growth_rate
        self.U5.add(transup_block(num_channels))
        self.U5.add(DenseBlock(numconvs_in_uppath[0], growth_rate, dropout_rate))

        num_channels = numconvs_in_uppath[0] * growth_rate
        self.U4.add(transup_block(num_channels))
        self.U4.add(DenseBlock(numconvs_in_uppath[1], growth_rate, dropout_rate))

        num_channels = numconvs_in_uppath[1] * growth_rate
        self.U3.add(transup_block(num_channels))
        self.U3.add(DenseBlock(numconvs_in_uppath[2], growth_rate, dropout_rate))

        num_channels = numconvs_in_uppath[2] * growth_rate
        self.U2.add(transup_block(num_channels))
        self.U2.add(DenseBlock(numconvs_in_uppath[3], growth_rate, dropout_rate))

        num_channels = numconvs_in_uppath[3] * growth_rate
        self.U1.add(transup_block(num_channels))
        self.U1.add(DenseBlock(numconvs_in_uppath[4], growth_rate, dropout_rate))

        self.final.add(nn.Conv2D(num_classes,kernel_size=1))
    def forward(self, x):
        for layer in self.P1:
            x = layer(x)
        p1=x
        for layer in self.P2:
            x=layer(x)
        p2 = x
        for layer in self.P3:
            x=layer(x)
        p3 = x
        for layer in self.P4:
            x=layer(x)
        p4 = x
        for layer in self.P5:
            x=layer(x)
        p5 = x
        for layer in self.P6:
            x=layer(x)
        for layer in self.U5:
            x=layer(x)
        x = nd.concat(x,p5)
        for layer in self.U4:
            x=layer(x)
        x = nd.concat(x, p4)

        for layer in self.U3:
            x = layer(x)
        x = nd.concat(x, p3)

        for layer in self.U2:
            x = layer(x)
        x = nd.concat(x, p2)

        for layer in self.U1:
            x = layer(x)
        x = nd.concat(x, p1)
        for layer in self.final:
            x=layer(x)
        return x
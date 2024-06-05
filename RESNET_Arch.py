from tensorflow.keras import layers,models
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Input, ReLU, Conv2DTranspose,Add
import tensorflow as tf 
from tensorflow_addons.layers import InstanceNormalization

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(ReflectionPadding2D, self).__init__()
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding, self.padding], [self.padding, self.padding], [0, 0]], mode='REFLECT')


class DownSampling(tf.keras.layers.Layer) :  

    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding='same',
        norm=True,
        lrelu=True,
    ):
        super(DownSampling, self).__init__()
        self.conv = layers.Conv2D(
            filters = out_channels,
            kernel_size = kernel_size,
            strides = stride, 
            padding = padding,
            use_bias = not norm
        )
        self.norm = layers.LayerNormalization() if norm else None
        self.activation = layers.LeakyReLU(0.2) if lrelu else layers.ReLU()


    def call(self, inputs, training=False) : 

        x = self.conv(inputs)

        if self.norm : 
            x = self.norm(x)

        x = self.activation(x)
        return x


class UpSampling(tf.keras.layers.Layer):

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding='same',
            output_padding=0,
            dropout=False,
        ):
            super(UpSampling, self).__init__()

            self.conv_t = layers.Conv2DTranspose(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                use_bias=False
            )

            self.norm = layers.LayerNormalization()
            self.dropout = layers.Dropout(0.5)  if dropout else None
            self.activation = layers.ReLU()
        
        def call(self, inputs, training=False) : 
             
            x = self.conv_t(inputs)
            x = self.norm(x)
            if self.dropout:
                x = self.dropout(x, training=training)
            x = self.activation(x)

            return x

class ResidualLayer(tf.keras.layers.Layer): 

    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(ResidualLayer, self).__init__()

        self.pad1 = ReflectionPadding2D(padding)
        self.downsampling = DownSampling(in_channels,in_channels,kernel_size, stride=1, padding = 'valid', lrelu= False)
        self.pad2 = ReflectionPadding2D(padding)
        self.downsampling2 = DownSampling(in_channels,in_channels,kernel_size, stride=1, padding = 'valid', lrelu= False)

    def call(self,inputs, training = False): 
        
        x = self.pad1(inputs)
        x = self.downsampling(x  )
        x = self.pad2(x)
        x= self.downsampling2(x )
        return inputs + x

### ResnetGenerator 

def Generator(res_blocks = 6, in_channels =3, hid_channels = 64, out_channels = 3): 

    inp = layers.Input(shape=(256, 256, in_channels))

    x = ReflectionPadding2D(3)(inp)
    x = DownSampling(in_channels, hid_channels, kernel_size=7, stride=1, padding='valid', lrelu=False)(x)  # 64x256x256
    x = DownSampling(hid_channels, hid_channels * 2, kernel_size=3, lrelu=False)(x)  # 128x128x128
    x = DownSampling(hid_channels * 2, hid_channels * 4, kernel_size=3, lrelu=False)(x)  # 256x64x64

    for _ in range(res_blocks):
        x = ResidualLayer(hid_channels * 4)(x)  # 256x64x64

    x = UpSampling(hid_channels * 4, hid_channels * 2, kernel_size=3, output_padding=1)(x)  # 128x128x128
    x = UpSampling(hid_channels * 2, hid_channels, kernel_size=3, output_padding=1)(x)  # 64x256x256

    x = ReflectionPadding2D(3)(x)
    x = layers.Conv2D(out_channels, kernel_size=7, strides=1, padding='valid')(x)  # 3x256x256
    x = layers.Activation('tanh')(x)

    return tf.keras.Model(inputs=inp, outputs=x)

def Discriminator(input_shape=(256, 256, 3),  in_channels =3, hid_channels = 64, out_channels = 3):
    inp = Input(shape=(256, 256, 3))

    x = DownSampling(in_channels, hid_channels, norm = False)(inp)  # 64x 128x128
    x = DownSampling(hid_channels, hid_channels * 2)(x)  # 128x128x128
    x = DownSampling(hid_channels * 2, hid_channels * 4,)(x)  # 256x64x64
    x = DownSampling(hid_channels*4,  hid_channels * 8,  stride=1)(x)
    x = layers.Conv2D (hid_channels * 8 , strides = 1, kernel_size = 4 , padding='same')(x)
    return tf.keras.Model(inputs=inp, outputs=x)

 
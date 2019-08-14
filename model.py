import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

## TODO
# - add args
# - use tf.data to fetch inputs
# - add import functionality
# - add export functionality
# - split forward & backward operations

class LeNet5():
    def __init__(self):
        # configs for run time session
        self.configProto = tf.ConfigProto(
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4
        )

        # initialize the computation graph
        self.graph = tf.Graph()

        # define input layer
        X = self.__input_layer()

        # define layer 1 (convolutional layer)
        C1 = self.__layer1(X)

        # define layer 2 (average pooling layer)
        S1 = self.__layer2(C1)

        # define layer 3 (convolutional layer)
        C2 = self.__layer3(S1)

        # define layer 4 (average pooling layer)
        S2 = self.__layer4(C2)

        # define layer 5 (convolutional layer)
        C3 = self.__layer5(S2)

        # define layer 6 (fully connected layer)
        FC1 = self.__layer6(C3)

        # define output layer (fully connected layer)
        FC2 = self.__output_layer(FC1)

        # define backprop
        self.__backprop_operations()
        
        with self.graph.as_default():
            self.saver = tf.train.Saver()

    @staticmethod
    def generate_variables(shape,name,zeros=False):
        if zeros:
            return tf.Variable(tf.zeros(shape), name=name)
        return tf.Variable(tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1), name=name)

    def __input_layer(self):
        with self.graph.as_default():
            return tf.placeholder(
                shape=(None,32,32,1),
                dtype=tf.float32,
                name="input_layer"
            )

    def __layer1(self, X):
        """
            LeNet-5 layer 1 (2d convolutional layer)
            ____________________________________
            kernel size     : (5,5) => [filter_height, filter_width, in_channels, out_channels]
            stride          : 1
            padding         : 0
            feature maps    : 6
            activation      : tanh
            ____________________________________
            input shape: (batch_size,32,32,1)  [batch, in_height, in_width, in_channels]
            output shape: (batch_size,28,28,6) [batch, out_height, out_width, out_channels]
            ____________________________________
        """
        # TODO: check X dimentions
        with self.graph.as_default():
            W = LeNet5.generate_variables(shape=(5,5,1,6), name="conv_layer_1_w")
            B = LeNet5.generate_variables(shape=(1,6), name="conv_layer_1_b", zeros=True)

            return tf.math.tanh( tf.nn.conv2d(X, W, strides=1, padding="VALID",name="conv_layer_1") + B , name="conv_layer_1_tanh")

                
    def __layer2(self, X):
        """
            LeNet-5 layer 2 (average pooling layer)
            ____________________________________
            kernel size: (2,2)
            stride          : 2
            padding         : 0
            feature maps    : 6
            ____________________________________
            input shape: (batch_size,28,28,6)
            output shape: (batch_size,14,14,6)
            ____________________________________
        """
        # TODO: check X dimentions
        with self.graph.as_default():
            return tf.nn.avg_pool(
                value=X,
                ksize=2,
                strides=2,
                padding="VALID",
                name="sample_layer_2"
            )
    
    def __layer3(self, X):
        """
            LeNet-5 layer 3 (2d convolutional layer)
            ____________________________________
            kernel size    : (5,5)
            stride         : 1
            padding        : 0
            feature maps   : 16
            activation     : tanh
            ____________________________________
            input shape: (batch_size,14,14,6)
            output shape: (batch_size,10,10,16)
            ____________________________________
            feature map connection between inputs
              | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
            0 | x |   |   |   | x | x | x |   |   | x | x  | x  | x  |    | x  | x  |
            1 | x | x |   |   |   | x | x | x |   |   | x  | x  | x  | x  |    | x  |
            2 | x | x | x |   |   |   | x | x | x |   |    | x  |    | x  | x  | x  |
            3 |   | x | x | x |   |   | x | x | x | x |    |    | x  |    | x  | x  |
            4 |   |   | x | x | x |   |   | x | x | x | x  |    | x  | x  |    | x  |
            5 |   |   |   | x | x | x |   |   | x | x | x  | x  |    | x  | x  | x  |
        """
        # TODO: currently connected all but it will be fixed later as mentioned upward
        # TODO: check X dimentions
        with self.graph.as_default():
            W = LeNet5.generate_variables(shape=(5,5,6,16), name="conv_layer_3_w")
            B = LeNet5.generate_variables(shape=(1,16), name="conv_layer_3_b", zeros=True)
            return tf.math.tanh( tf.nn.conv2d(X, W, strides=1, padding="VALID", name="conv_layer_3") + B , name="conv_layer_3_tanh")

    
    def __layer4(self, X):
        """
            LeNet-5 layer 4 (average pooling layer)
            ____________________________________
            kernel size     : (2,2)
            stride          : 2
            padding         : 0
            feature maps    : 16
            ____________________________________
            input shape: (batch_size,10,10,16)
            output shape: (batch_size,5,5,16)
            ____________________________________
        """
        # TODO: check X dimentions
        with self.graph.as_default():
            return tf.nn.avg_pool(
                value=X,
                ksize=2,
                strides=2,
                padding="VALID",
                name="sample_layer_4"
            )

    def __layer5(self, X):
        """
            LeNet-5 layer 5 (2d convolutional layer)
            ____________________________________
            kernel size     : (5,5)
            stride          : 1
            padding         : 0
            feature maps    : 120
            activation      : tanh
            ____________________________________
            input shape: (batch_size,5,5,16)
            output shape: (batch_size,1,1,120)
            ____________________________________
        """
        # TODO: check X dimentions
        with self.graph.as_default():
            W = LeNet5.generate_variables(shape=(5,5,16,120), name="conv_layer_5_w")
            B = LeNet5.generate_variables(shape=(1,120), name="conv_layer_5_b", zeros=True)
            return tf.math.tanh( tf.nn.conv2d(X, W, strides=1, padding="VALID", name="conv_layer_5") + B , name="conv_layer_5_tanh")
    
    def __layer6(self, X):
        """
            LeNet-5 layer 6 (fully connected layer)
            ____________________________________
            neuron count                : 84
            feature count               : 120
            bias count                  : 84 (1 for each neuron)
            total trainable parameters  : 84*120+84 = 10164
            activation                  : tanh
            ____________________________________
            input shape: (batch_size,1,1,120) => reshape (batch_size,120)
            output shape: (batch_size,84)
            ____________________________________
        """
        # TODO: check X dimentions
        with self.graph.as_default():
            W = LeNet5.generate_variables(shape=(120,84), name="fc_layer_6_w")
            B = LeNet5.generate_variables(shape=(1,84), name="fc_layer_6_b", zeros=True)
            
            return tf.math.tanh( tf.matmul(tf.reshape(X, (-1,120)), W) + B , name="fc_layer_6_tanh")
    
    def __output_layer(self, X):
        """
            LeNet-5 layer 7 (fully connected layer)
            ____________________________________
            neuron count                : 10
            feature count               : 84
            bias count                  : 10 (1 for each neuron)
            total trainable parameters  : 10*84+10 = 850
            activation                  : softmax (it will be applied in training due to performance issue)
            ____________________________________
            input shape: (batch_size,84)
            output shape: (batch_size,10)
            ____________________________________
        """
        # TODO: check X dimentions
        with self.graph.as_default():
            W = LeNet5.generate_variables(shape=(84,10), name="output_layer_w")
            B = LeNet5.generate_variables(shape=(1,10), name="output_layer_b", zeros=True)
            return tf.add(tf.matmul(X, W),B, name="output_layer")

    def __backprop_operations(self, num_of_classes=10, learning_rate=0.001):# TODO add num_of_classes
        with self.graph.as_default():
            # define Y tensor for actual results
            Y = tf.placeholder(shape=(None,num_of_classes), dtype=tf.float32, name="Y_TRUE")

            # define loss
            logits = self.graph.get_tensor_by_name("output_layer:0")
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits, name="loss")
            
            # define cost
            cost = tf.reduce_mean(loss,name="cost")

            # define optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # minimize cost
            optimizer.minimize(cost, name="train")

            # define accuracy metrics
            matches = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
            tf.reduce_mean(tf.cast(matches,tf.float32), name="accuracy")

    def __evaluate(self, X_train, Y_train, X_test, Y_test, epoch, iteration):
        sess = tf.get_default_session()
        X = sess.graph.get_tensor_by_name("input_layer:0")
        Y = sess.graph.get_tensor_by_name("Y_TRUE:0")
        cost = sess.graph.get_tensor_by_name("cost:0")
        acc = sess.graph.get_tensor_by_name("accuracy:0")

        # evaluate
        test_stats = sess.run((cost,acc), feed_dict={X:X_test, Y:Y_test} )

        train_stats = sess.run((cost,acc), feed_dict={X:X_train, Y:Y_train} )
        
        verbose1 = "EPOCH: {}/{}\tIteration: {}/{}\n".format(epoch[0],epoch[1],iteration[0],iteration[1])
        verbose2 = "Test:\ncost: {:.3f}\taccuracy: {:.3f}\n".format(test_stats[0],test_stats[1])
        verbose3 = "Train:\ncost: {:.3f}\taccuracy: {:.3f}\n".format(train_stats[0],train_stats[1])
        wrapper = "_______________________________________\n"
        return wrapper+verbose1+verbose2+verbose3+wrapper

    @staticmethod
    def train_test_split(X, Y, train_portion):
        assert X.shape[0] == Y.shape[0]
        assert train_portion > 0 and train_portion <= 1

        randomize = np.arange(X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        break_point = int(X.shape[0]*train_portion)
        
        # X_train,Y_train,X_test,Y_test 
        return X[:break_point], Y[:break_point], X[break_point:], Y[break_point:]


    def train(self, data, train_portion=0.8, batch_size=128, learning_rate=0.001, epoch=4):
        """
            data as tuple with format of:
            (
                (data_set_size,1024), # images as numpy ndarray
                (data_set_size,10)    # labels as numpy ndarray
            )
        """
        X_train,Y_train,X_test,Y_test = LeNet5.train_test_split(data[0],data[1],train_portion)
        del data
        
        num_of_classes = Y_train.shape[1]
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]

        X_train = X_train.reshape(train_size,32,32,1)
        X_test = X_test.reshape(test_size,32,32,1)

        # forward and backward prop
        
        with tf.Session(graph=self.graph, config=self.configProto) as sess:
            # initialize graph variables
            init = tf.global_variables_initializer()
            gradient = self.graph.get_operation_by_name("train")
            cost = self.graph.get_tensor_by_name("cost:0")
            acc = self.graph.get_tensor_by_name("accuracy:0")
            X = self.graph.get_tensor_by_name("input_layer:0")
            Y = self.graph.get_tensor_by_name("Y_TRUE:0")
            print("session started...")

            # initialize data
            sess.run(init)
            for i in range(epoch):
                # shuffle the train set for every epoch
                randomize = np.arange(train_size)
                np.random.shuffle(randomize)

                X_train = X_train[randomize]
                Y_train = Y_train[randomize]

                for offset in range(0,train_size-batch_size,batch_size):

                    # forward-backward prop
                    sess.run(gradient, feed_dict={X: X_train[offset:offset+batch_size] , Y:Y_train[offset:offset+batch_size]})
                    if offset % 10 == 0:
                        verbose = self.__evaluate(
                            X_train[offset:offset+batch_size],Y_train[offset:offset+batch_size],
                            X_test,Y_test,
                            (i+1,epoch),
                            (offset+batch_size,train_size)
                        )
                        print(verbose)
                    time.sleep(0.1)
                self.saver.save(sess,"lenet")

if __name__ == '__main__':
    # create the lenet-5 model
    model = LeNet5()

    # download the mnist data
    mnist = input_data.read_data_sets("data/", one_hot=True)
    image_data = np.append(mnist.train.images, mnist.test.images, axis=0)
    label_data = np.append(mnist.train.labels, mnist.test.labels, axis=0)
    del mnist
    # convert 28x28 images to 32x32 images since LeNet-5 designed to handle 32x32 gray scale images
    image_data = np.pad(image_data.reshape(image_data.shape[0],28,28),((0,0),(2,2),(2,2)),"constant").reshape(image_data.shape[0],1024)
    
    model.train(data=(image_data,label_data), epoch=4, learning_rate=0.01, batch_size=512)

    


import tensorflow as tf
import numpy as np

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
	
# 1.训练的数据
# Make up some real data
def init_x_data(trainset):
    x_data = np.array(trainset);
    return x_data;

def init_y_data():
    y_data = np.zeros((5,3395),dtype=int)
    for i in range(74):
        y_data[0,i]=1
    for i in range(74,1942+74):
        y_data[1,i]=1
    for i in range(74,1942+74+555):
        y_data[2,i]=1
    for i in range(74,1942+74+555+285):
        y_data[3,i]=1
    for i in range(74,1942+74+555+285+539):
        y_data[4,i]=1
    return y_data

def train(x_data,y_data):
    # 2.定义节点准备接收数据
    xs = tf.placeholder(tf.float32,[None,3])
    ys = tf.placeholder(tf.float32,[None,5])
    # 3.定义神经层：隐藏层和预测层
    l1 = add_layer(xs, 3, 10, activation_function=tf.nn.relu)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = add_layer(l1, 10, 5, activation_function=None)
    # 4.定义 loss 表达式
    # the error between prediciton and real data    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
    # 5.选择 optimizer 使 loss 达到最小                   
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1       
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # important step 对所有变量进行初始化
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)
    # 迭代 1000 次学习，sess.run optimizer
    for i in range(3000):
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    result = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
    #将所有的 variable 保存到定义的路径
    save_path = saver.save(sess,"train/variables.ckpt")
    sess.close()
    return result

def intergration(network_path):
    #从指定的路径中将variables取出来
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess,network_path)
    return sess;

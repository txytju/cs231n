# Different CNN architectures defined in this file.
# Input image size is [32, 32, 3]

# [conv-relu-pool]x3 -> [affine]x2 -> [softmax]
def my_model_1(X,y,is_training):
    
    # conv-relu-pool 1
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    # conv-relu-pool 2
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    # conv-relu-pool 3
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 128])
    bconv3 = tf.get_variable("bconv3", shape=[128])
    
    a1 = tf.nn.conv2d(X, Wconv1, [1,1,1,1], padding="SAME") + bconv1
    a2 = tf.nn.relu(a1)
    a3 = tf.nn.max_pool(a2, [1,2,2,1], [1,2,2,1], padding="SAME")
    
    a4 = tf.nn.conv2d(a3, Wconv2, [1,1,1,1], padding="SAME") + bconv2
    a5 = tf.nn.relu(a4)
    a6 = tf.nn.max_pool(a5, [1,2,2,1], [1,2,2,1], padding="SAME")
    
    a7 = tf.nn.conv2d(a6, Wconv3, [1,1,1,1], padding="SAME") + bconv3
    a8 = tf.nn.relu(a7)
    a9 = tf.nn.max_pool(a8, [1,2,2,1], [1,2,2,1], padding="SAME")
    
    
    
    a9_reshape = tf.reshape(a9, [-1,2048])
    a10 = tf.layers.dense(a9_reshape, 1024, activation=tf.nn.relu)
    a11 = tf.layers.dense(a10, 100, activation=tf.nn.relu)
    a12 = tf.layers.dense(a11, 10)
    
    return a12



def my_model_2(X,y,is_training):
    
    # conv-relu-pool 1
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    # conv-relu-pool 2
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    # conv-relu-pool 3
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 128])
    bconv3 = tf.get_variable("bconv3", shape=[128])
    # conv-relu-pool 4
    Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 128, 256])
    bconv4 = tf.get_variable("bconv4", shape=[256])
    
    a1 = tf.nn.conv2d(X, Wconv1, [1,1,1,1], padding="SAME") + bconv1
    a1 = tf.nn.relu(a1)
    a2 = tf.nn.max_pool(a1, [1,2,2,1], [1,2,2,1], padding="SAME")
    
    a3 = tf.nn.conv2d(a2, Wconv2, [1,1,1,1], padding="SAME") + bconv2
    a3 = tf.nn.relu(a3)
    a4 = tf.nn.max_pool(a3, [1,2,2,1], [1,2,2,1], padding="SAME")
    
    a5 = tf.nn.conv2d(a4, Wconv3, [1,1,1,1], padding="SAME") + bconv3
    a5 = tf.nn.relu(a5)
    a6 = tf.nn.max_pool(a5, [1,2,2,1], [1,2,2,1], padding="SAME")
    
       
    a6_reshape = tf.reshape(a6, [-1,1024])
    a10 = tf.layers.dense(a6_reshape, 500, activation=tf.nn.relu)
    a11 = tf.layers.dense(a10, 100, activation=tf.nn.relu)
    a12 = tf.layers.dense(a11, 10)
    
    return a12
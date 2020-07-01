import numpy as np
import tensorflow as tf

def read_file():
    filename = "./data.txt"
    with open(filename, 'r+') as f:
        s = [list(map(float,i[:-1].split(','))) for i in f.readlines()]
    # print(s)
    dataset = []
    yset = []
    for i in s:
        dataset.append(i[:-1])
        yset.append(i[-1])
    dataset_array = np.array(dataset)
    yset_array = np.array(yset)
    yset_array = yset_array.reshape((-1,1))
    print(dataset_array.shape)
    print(yset_array.shape)
    return dataset_array, yset_array
    

learning_rate = 0.01
model_path = "./model/"  
log_dir = "./log/"

def gyro_nn(train, dataset, yset):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 2], name = 'x_input')
        y = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

#     def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
#         with tf.name_scope(layer_name):
#             with tf.name_scope('weights', reuse=tf.AUTO_REUSE):
#                 weights = weight_variable([input_dim, output_dim])
#             with tf.name_scope('bias',reuse=tf.AUTO_REUSE):
#                 bias = bias_variable([output_dim])
#             with tf.name_scope('linear_compute'):
#                 preactivate = tf.matmul(input_tensor, weights) + bias
#             activations = act(preactivate, name='activation')
#             return activations
        
    def nn_layer(input_tensor, input_dim, output_dim, act = tf.nn.relu):
        weights = weight_variable([input_dim, output_dim])
        bias = bias_variable([output_dim])
        preactivate = tf.matmul(input_tensor, weights) + bias
        activations = act(preactivate, name='activation')
        return activations

    hidden1 = nn_layer(x, 2, 60)
    hidden2 = nn_layer(hidden1, 60, 30)
    predict = nn_layer(hidden2, 30, 1)

    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(y, predict)
    tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + "train/", sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + "test/")
        if train:
            print('training mode')
            sess.run(tf.global_variables_initializer())
            train_feed_dict = {
                x: dataset,
                y: yset
            }
            for step in range(150000):
                if(step % 100 == 0):
                    _, loss_val, summary = sess.run([train_op, loss, merged],
                                                    feed_dict=train_feed_dict)
                    print("step = {}\tloss = {}".format(step, loss_val))
                elif(step % 100 == 0):
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, loss_val, summary = sess.run([train_op, loss, merged],
                                                    feed_dict=train_feed_dict,
                                                   options=run_options,
                                                   run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    print('Adding run metadata for ', step)
                else:
                    _, loss_val, summary = sess.run([train_op, loss, merged],
                                                    feed_dict=train_feed_dict)
                    train_writer.add_summary(summary, step)
            train_writer.close()
            test_writer.close()
            saver.save(sess, model_path)
            print("end of training, save model to {}".format(model_path))
        else:
            print('test mode')
            saver.restore(sess, model_path)
            print("load model from {}".format(model_path))
            test_feed_dict = {
                x: dataset,
                #y: yset
            }
            prediction = sess.run(predict,feed_dict=test_feed_dict)
            print('input is: ', dataset)
            print('output is: ', prediction)
            return prediction
            
if __name__ == "__main__":
    dataset_array, yset_array = read_file()
    dataset = dataset_array / dataset_array.max(axis=0)
    #y_set = yset_array / yset_array.max()
    y_set = yset_array / 1000.0
    print('data shape: ',dataset.shape)
    print('y shape: ',y_set.shape)
    train = True
    gyro_nn(train, dataset, y_set)
    #gyro_nn(train, dataset, yset_array)
import time
from ops import *
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split

class Resnet(object):
    def __init__(self, sess, args):
        self.model_name = args.name
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.dataset_dir = args.image_dir
        self.label_dir = args.label_dir
        self.dataset = np.load(self.dataset_dir)
        self.label = np.load(self.label_dir)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.dataset, self.label, test_size=0.2)
        self.h_img_size = self.dataset.shape[1]
        self.w_img_size = self.dataset.shape[2]
        self.c_dim = self.dataset.shape[3]
        self.label_dim = self.label.shape[1]
        self.res_n = args.res_n
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x)//self.batch_size
        self.init_lr = args.lr
        self.log_dir = args.log_dir




    #CONSTRUCTION

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope('network', reuse=reuse):

            if self.res_n < 50:
                residual_block = resblock

            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            #First phase
            ch =64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]):
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='Firstphase_'+str(i))

            #Second phase

            x = residual_block(x, channels = ch*2, is_training=is_training, downsample= True, scope = 'Secondphase_0')

            for i in range(1, residual_list[1]):
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope = 'Secondphase_'+str(i))

            #Third_phase

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope = 'Thirdphase_0')

            for i in range(1, residual_list[2]):
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False , scope = 'Thirphase_'+str(i))

            #Fourth_phase

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope = 'Fourthphase_0')

            for i in range(1, residual_list[3]):
                x = residual_block(x, channels=ch*8 , is_training=is_training, downsample=False, scope='Fourthphase_'+str(i))

            x = batch_norm(x, is_training=is_training, scope = 'batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_connected(x,units = self.label_dim, scope = 'logit')

            return x



    #Model

    def build_model(self): #Height size = h_img_size , Width size = w_img_size , Color dimension = c_dim
        self.train_inputs = tf.placeholder(tf.float32, [self.batch_size, self.h_img_size, self.w_img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name = 'train_labels')

        self.test_inputs = tf.placeholder(tf.float32, [None, self.h_img_size, self.w_img_size, self.c_dim], name = 'test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [None, self.label_dim], name = 'test_labels')

        self.lr = tf.placeholder(tf.float32, name = 'learning_rate')

        ##network
        self.train_logits = self.network(self.train_inputs)
        self.test_logits = self.network(self.test_inputs, is_training = False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(logit = self.train_logits, label = self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(logit = self.test_logits, label = self.test_labels)


        ##training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)

        ##summary
        self.summary_train_loss = tf.summary.scalar('train_loss', self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar('train_accuracy', self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar('test_loss', self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar('test_accuracy', self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])


    #Train

    def train(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

        self.writer = tf.summary.FileWriter(self.log_dir+'/'+self.model_dir, self.sess.graph)

        #restore if exist
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch > int(self.epoch *0.75):
                epoch_lr = epoch_lr * 0.01
            elif start_epoch > int(self.epoch *0.5) and start_epoch < int(self.epoch *0.75):
                epoch_lr = epoch_lr *0.1
            print('Load success')

        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter =1
            print('Load failed')

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch *0.5) or epoch ==int(self.epoch *0.75):
                epoch_lr = epoch_lr *0.1

            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_x = data_augmentation(batch_x, self.h_img_size, self.w_img_size)


                rand_idx = np.random.randint(low=0, high=len(self.test_x), size=self.batch_size)
                batch_x_test = self.test_x[rand_idx]
                batch_y_test = self.test_y[rand_idx]

                train_feed_dict = {self.train_inputs : batch_x, self.train_labels : batch_y, self.lr : epoch_lr}
                test_feed_dict = {self.test_inputs : batch_x_test, self.test_labels : batch_y_test}


                #Update network
                _, summary_str, train_loss, train_accuracy = self.sess.run([self.optimizer, self.train_summary, self.train_loss, self.train_accuracy],
                                                                           feed_dict=train_feed_dict)

                #Test
                summary_str, test_loss, test_accuracy = self.sess.run([self.test_summary, self.test_loss, self.test_accuracy],
                                                                      feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                counter +=1
                print(
                    "Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                    % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            start_batch_id =0

            self.save(self.checkpoint_dir, counter)

        self.save(self.checkpoint_dir, counter)


    @property
    def model_dir(self):
        return '{}{}_{}_{}'.format(self.model_name, self.res_n, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)


    def load(self, checkpoint_dir):
        print('Reading')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess , os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print('Find checkpoint ->  {} succeed'.format(ckpt_name))
            return True, counter
        else:
            print('Failed to find a checkpoint')
            return False, 0


    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print('Load success')

        else:
            print('Load failed')

        rand_idx = np.random.randint(low=0, high=len(self.test_x), size=self.batch_size)
        test_feed_dict ={self.test_inputs : self.test_x[rand_idx], self.test_labels : self.test_y[rand_idx]}
        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print('Test_accuracy :{}'.format(test_accuracy))
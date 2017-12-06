import tensorflow as tf
import numpy as np

n_hidden_1=300
n_hidden_2=100
num_iter=10000

data=np.genfromtxt("MNIST_data/train.csv",delimiter=',')
X_=data[1:5000,1:]
Y_=data[1:5000,0]
X_test=data[7000:8000,1:]
Y_test=data[7000:8000,0]
Y_vec=np.zeros((Y_.shape[0],10))
for i in range(0,Y_.shape[0]):
	Y_vec[i]=np.array([(1 if j==Y_[i] else 0) for j in range(0,10)])
Y_=Y_vec
Y_vec2=np.zeros((Y_test.shape[0],10))
for i in range(0,Y_test.shape[0]):
	Y_vec2[i]=np.array([(1 if j==Y_test[i] else 0) for j in range(0,10)])
Y_test=Y_vec2	

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

W1=tf.Variable(tf.random_normal([784,n_hidden_1],stddev=0.1))
b1=tf.Variable(tf.constant(0.1,shape=[n_hidden_1]))
W2=tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=0.1))
b2=tf.Variable(tf.constant(0.1,shape=[n_hidden_2]))
W3=tf.Variable(tf.random_normal([n_hidden_2,10],stddev=0.1))
b3=tf.Variable(tf.constant(0.1,shape=[10]))

layer1=tf.add(tf.matmul(X,W1),b1)
layer1=tf.nn.relu(layer1)
layer2=tf.add(tf.matmul(layer1,W2),b2)
layer2=tf.nn.relu(layer2)
out=tf.add(tf.matmul(layer2,W3),b3)
# out=tf.nn.sigmoid(layer3)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=out))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cost)

correct_prediction=tf.equal(tf.argmax(out,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

# file_writer=tf.summary.FileWriter('./Graph',sess.graph)
tf.summary.scalar("cost",cost)
summary_op=tf.summary.merge_all()
writer=tf.summary.FileWriter('./Graph',sess.graph)	

for i in range(0,num_iter):
	sess.run(train_step,feed_dict={X:X_,Y:Y_})
	summary=sess.run(summary_op,feed_dict={X:X_,Y:Y_})
	writer.add_summary(summary,i)
	if(i%50==0):
		print i
	if(i%100==0):
		print(accuracy.eval(feed_dict={X:X_test,Y:Y_test}))
saver=tf.train.Saver([W1,b1,W2,b2,W3,b3])
save_path=saver.save(sess,"/home/sanskar/DL/mnist/restore.ckpt")

print("Trained")
print("Saved in "+save_path)
import tensorflow as tf

state = tf.Variable(0)

one = tf.constant(1)

newValue = tf.add(state,one)

update = tf.assign(state,newValue)

initOp = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(initOp)
    print(session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))

a = tf.placeholder(tf.float32)

b = a*2

with tf.Session() as session:
    result = session.run(b,feed_dict={a:3.5})
    print(result)
    
a = tf.placeholder(tf.float32)

b = a*2

with tf.Session() as session:
    result = session.run(b,feed_dict={a:[ [[6, 7, 8], [9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20], [21, 22, 23]] , [[15, 16, 17], [18, 19, 20], [21, 22, 23]] ]})
    print(result)
    


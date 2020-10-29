


 Class Func:
    def __init__(self, out_dir, lat_dim, trans):
        self.out_dir = out_dir
        self.lat_dim = lat_dim
        self.trans = trans

    def mvn_pdf(x, mu=0, sig=1):
        return tf.exp(tf.reduce_sum((x - mu)**2, axis=1) / sig**2) / (2 * np.pi * sig**2)**(x.shape[1] / 2)

    def d_z_prob(z, z_, gam):
        pz = mvn_pdf(z_, z, gam)
        p0 = mvn_pdf(z_)
        return pz / (pz + p0)

    def mh_update(prev, gam, sig=1):
        cand = prev + tf.random.normal(prev.shape, mean=0.0, stddev=gam)
        p = tf.minimum(1.0, tf.math.exp(tf.reduce_sum(prev**2 - cand**2, axis=1) / sig**2)) # tf.minumum unnecessary
        u = tf.random.uniform(p.shape)
        return tf.where(tf.expand_dims(u < p, axis=1), cand, prev)

    def distance2(z):
        r = tf.reduce_sum(z*z, -1)
        D = tf.reshape(r, [-1, 1]) - 2*tf.matmul(z, z, transpose_b=True) + tf.reshape(r, [1, -1])
        return D

    def trans_random(x, gam):
        e = tf.random.normal([x.shape[0], lat_dim], 0, gam)
        return self.trans([x, e], training=False)

    def plot_images(model, epoch, sample_input, n_steps, n_examples, gam):
        tf.random.set_seed(10)
        generated_images = model(sample_input, gam)

        fig = plt.figure(figsize=(n_examples, 1.1 * n_steps))
        for j in range(n_examples):
            plt.subplot(n_steps+1, n_examples, j + 1)
            plt.imshow(tf.squeeze(sample_input[j]) * 127.5 + 127.5, cmap='gray')
            plt.axis('off')  

            plt.subplot(n_steps+1, n_examples, n_examples + j + 1)
            plt.imshow(tf.squeeze(generated_images[j]) * 127.5 + 127.5, cmap='gray')
            plt.axis('off')   

        plt.savefig(os.path.join(self.out_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.close(fig)   

import logging
import os
import shutil
import threading

import tensorflow as tf

from test_tips.test_input_data import load_data_sets
from test_tips.test_cfl import construct_model
from test_tips.test_ops import dist_ae_transformer
from test_tips.test_ops import dist_normalizer
from test_tips.test_ops import dist_transformer
from test_tips.test_utils import dist_check_args
from test_tips.test_utils import dist_parser
from test_tips.test_utils import load_model, reduce_product

logger = logging.getLogger(__name__)


def enqueue(coord,
            sess,
            aux,
            batch_size,
            unlabeled=False,
            return_labels=False,
            directed=False):
    (enqueue_op, queues, data) = aux

    while not coord.should_stop():
        if unlabeled:
            if directed:
                batch_src = data.next_source_batch(
                    batch_size, return_labels=return_labels)
                batch_dst = data.next_target_batch(
                    batch_size, return_labels=return_labels)
                batch = batch_src + batch_dst
            else:
                batch = data.next_unlabeled_batch(
                    batch_size, return_labels=return_labels)
                batch *= 2
        else:
            batch = data.next_batch(batch_size)
        assert len(queues) == len(batch), '{} != {}'.format(
            len(queues), len(batch))
        feed_dict = {queue: subbatch for queue, subbatch in zip(queues, batch)}
        sess.run(enqueue_op, feed_dict=feed_dict)


def train_loop(sess, model, data, aux, batch_size, start_iter, epochs,
               post_epochs, eval_epochs, save_iters, disable_eval, log_dir,
               checkpoint_dir, saver, directed):

    best_saver = tf.train.Saver()
    best_acc_saver = tf.train.Saver()
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # enqueue train
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[0], batch_size])
    enqueue_thread.daemon = True
    enqueue_thread.start()

    # enqueue unlabeled
    kwargs = {'unlabeled': True, 'directed': directed}
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[1], batch_size], kwargs=kwargs)
    enqueue_thread.daemon = True
    enqueue_thread.start()

    # enqueue val
    enqueue_thread = threading.Thread(
        target=enqueue, args=[coord, sess, aux[2], batch_size])
    enqueue_thread.daemon = True
    enqueue_thread.start()

    best_dir = os.path.join(checkpoint_dir, 'best_model')
    os.makedirs(best_dir, exist_ok=True)
    best_acc_dir = os.path.join(checkpoint_dir, 'best_acc_model')
    os.makedirs(best_acc_dir, exist_ok=True)
    try:
        model.train(
            sess=sess,
            data=data,
            start_iter=start_iter,
            epochs=epochs,
            post_epochs=post_epochs,
            best_dir=best_dir,
            best_acc_dir=best_acc_dir,
            checkpoint_dir=checkpoint_dir,
            eval_epochs=eval_epochs,
            disable_eval=disable_eval,
            saver=saver,
            best_saver=best_saver,
            best_acc_saver=best_acc_saver,
            save_iters=save_iters,
            writer=writer)
    finally:
        coord.request_stop()
        coord.join(threads)


def train_dist(load_pre_weights, data_name, data_root, checkpoint_root,
               data_switch, log_root, run_tag, seed, source_shape, input_shape,
               ae_shape, pos_weight, batch_size, data_scale, data_mean,
               data_norm, data_type, data_mirror, data_random_crop,
               data_is_image, data_is_double, data_disable_double, latent_norm,
               latent_shape, model_type, gan_type, num_components, latent_size,
               raw_latent, caffe_margin, gan, cgan, t_dim, dist_type, act_type,
               use_threshold, lr, beta1, beta2, z_dim, z_stddev, g_dim, g_lr,
               g_beta1, g_beta2, m_prj, m_enc, d_dim, d_lr, d_beta1, d_beta2,
               lambda_dra, lambda_gp, lambda_m, directed, data_directed,
               reg_const, epochs, post_epochs, eval_epochs, save_iters,
               disable_eval, reset):
    tf.set_random_seed(seed)

    input_size = reduce_product(input_shape)
    source_size = reduce_product(source_shape) if source_shape else input_size


    print("\n--------------------------------Main Paths------------------------------------")

    data_dir = os.path.join(data_root, data_name)
    checkpoint_dir = os.path.join(checkpoint_root, data_name)
    log_dir = os.path.join(log_root, data_name)
    print('data_dir:{}'.format(data_dir))
    print('checkpoint_dir:{}'.format(checkpoint_dir))
    print('log_dir:{}'.format(log_dir))

    # load data sets
    data = load_data_sets(
        data_dir,
        source_size,
        is_image=data_is_image,
        is_double=data_is_double,
        raw_latent=raw_latent,
        directed=directed or data_directed,
        data_switch=data_switch,
        seed=seed)

    print('------------------------Dist Transformer-----------------')

    print('source_shape:{}'.format(source_shape))
    print('input_shape:{}'.format(input_shape))
    print('data_random_crop:{}'.format(data_random_crop))
    print('data_mirror:{}'.format(data_mirror))
    print('ae_shape:{}'.format(ae_shape))
    #Minist:source_shape:None; input_shape:(28, 28, 1); data_random_crop:False; data_mirror:False; ae_shape:None
    #Minist:train_data_transformer(tensor) =  tf.reshape(tensor, (-1, ) + tuple((28,28,1))); val_data_transformer(tensor) = tf.reshape(tensor, (-1, ) + tuple((28,28,1))); ae_transformer (tensor) = tensor
    train_data_transformer, val_data_transformer = dist_transformer(
        source_shape=source_shape,
        input_shape=input_shape,
        data_random_crop=data_random_crop,
        data_mirror=data_mirror)

    ae_transformer = dist_ae_transformer(
        input_shape=input_shape, ae_shape=ae_shape)


    print('------------------------Dist Normalizer-----------------')

    print('data_scale:{}'.format(data_scale))
    print('data_mean:{}'.format(data_mean))
    print('data_norm:{}'.format(data_norm))
    print('latent_norm:{}'.format(latent_norm))
    print('data_type:{}'.format(data_type))


    #Minist:data_normalizer(tensor) =  tf.reshape(tf.clip_by_value(tensor, 0, 1), (-1,784)); data_unnormalizer(tensor) = tf.reshape(tensor, (-1,28,28,1)); ae_transformer = data_normalizer; ae_unnormalizer = data_unnormalizer; latent_normalizer = None

    (data_normalizer, data_unnormalizer, ae_normalizer, ae_unnormalizer,
     latent_normalizer) = dist_normalizer(
         input_shape=input_shape,
         ae_shape=ae_shape,
         data_scale=data_scale,
         data_mean=data_mean,
         data_norm=data_norm,
         latent_norm=latent_norm,
         data_type=data_type)

    print('------------------------Construct Model----------------------')

    #Minist: is_double=false, disable_double=false, source_shape=None, input_shape=(28,28,1), ae_shape=None, pos_weight=None, latent_shape=None, batch_size=100, data_norm=None, data_type='Sigmoid', model_type='conv', gan_type='conv', num_components=2/3/4/5, latent_size=20/15/12/10, caffe_margin=None, gan=false, cgan=false, t_dim=None,dist_type='pcd'/'monomer'/'siamese', act_type=none, use_threshold=true, lr=0.001, beta1=0.9, beta2=0.999, z_dim=20, z_stddev=1.0, g_dim=64, g_lr=0.0002, g_beta1=0.5, g_beta2=0.999, m_prj=none, m_enc=none, d_dim=64, d_lr=0.0002, d_beta1=0.5, d_beta2=0.999, lambda_gp=none, lambda_dra=0.5, lambda_m=0.0, directed=false, data_directed=false, reg_const=0.0005, data, run_tag=none, train_data_transformer, val_data_transformer, ae_transformer, data_normalizer, data_unnormalizer, ae_normalizer, ae_unnormalizer, latent_normalizer, enable_input_producer = true;


    model, aux = construct_model(
        is_double=data_is_double,
        disable_double=data_disable_double,
        latent_shape=latent_shape,
        source_shape=source_shape,
        input_shape=input_shape,
        ae_shape=ae_shape,
        batch_size=batch_size,
        data_norm=data_norm,
        data_type=data_type,
        model_type=model_type,
        gan_type=gan_type,
        num_components=num_components,
        latent_size=latent_size,
        pos_weight=pos_weight,
        caffe_margin=caffe_margin,
        gan=gan,
        cgan=cgan,
        t_dim=t_dim,
        dist_type=dist_type,
        act_type=act_type,
        use_threshold=use_threshold,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        z_dim=z_dim,
        z_stddev=z_stddev,
        g_dim=g_dim,
        g_lr=g_lr,
        g_beta1=g_beta1,
        g_beta2=g_beta2,
        m_prj=m_prj,
        m_enc=m_enc,
        d_dim=d_dim,
        d_lr=d_lr,
        d_beta1=d_beta1,
        d_beta2=d_beta2,
        lambda_dra=lambda_dra,
        lambda_gp=lambda_gp,
        lambda_m=lambda_m,
        directed=directed,
        data_directed=data_directed,
        reg_const=reg_const,
        data=data,
        run_tag=run_tag,
        train_data_transformer=train_data_transformer,
        val_data_transformer=val_data_transformer,
        ae_transformer=ae_transformer,
        data_normalizer=data_normalizer,
        data_unnormalizer=data_unnormalizer,
        ae_normalizer=ae_normalizer,
        ae_unnormalizer=ae_unnormalizer,
        latent_normalizer=latent_normalizer,
        enable_input_producer=True)

    # get paths
    no_gan_checkpoint_dir = os.path.join(
        checkpoint_dir,
        model.get_name(no_gan=True)) if load_pre_weights else None
    checkpoint_dir = os.path.join(checkpoint_dir, model.get_name())
    log_dir = os.path.join(log_dir, model.get_name())

    print('-------------------------------------Get Paths------------------------------------')
    print('no_gan_checkpoint_dir:{}'.format(no_gan_checkpoint_dir))
    print('checkpoint_dir:{}'.format(checkpoint_dir))
    print('log_dir:{}'.format(log_dir))

    # reset files
    if reset:
        for path in [checkpoint_dir, log_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)

    # create files
    for path in [checkpoint_dir, log_dir]:
        os.makedirs(path, exist_ok=True)

    # save logging to text file
    log_format = '%(asctime)s [%(levelname)-5.5s] [%(name)s]  %(message)s'
    logging.basicConfig(
        filename=os.path.join(log_dir, 'log.log'),
        format=log_format,
        level=logging.WARNING)

    rootLogger = logging.getLogger()
    formatter = logging.Formatter(log_format)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    rootLogger.addHandler(consoleHandler)
    logger.warning('run with %s', model.get_name())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    logger.info('start session')
    with tf.Session(config=config) as sess:
        saver, start_iter = load_model(sess, checkpoint_dir,
                                       no_gan_checkpoint_dir)
        train_loop(
            sess=sess,
            model=model,
            aux=aux,
            data=data,
            batch_size=batch_size,
            start_iter=start_iter,
            epochs=epochs,
            post_epochs=post_epochs,
            eval_epochs=eval_epochs,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            disable_eval=disable_eval,
            save_iters=save_iters,
            saver=saver,
            directed=directed or data_directed)


def main():
    args = parse_args()
    train_dist(**vars(args))


def parse_args():
    parser = dist_parser(batch_size=100)
    parser.add_argument('--load-pre-weights', action='store_true')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--save-iters', type=int)
    parser.add_argument('--data-switch', action='store_true')
    parser.add_argument('--post-epochs', type=int, default=100)
    parser.add_argument('--eval-epochs', type=int, default=1)
    parser.add_argument('--disable-eval', action='store_true')
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()
    dist_check_args(args)
    return args


if __name__ == '__main__':
    main()

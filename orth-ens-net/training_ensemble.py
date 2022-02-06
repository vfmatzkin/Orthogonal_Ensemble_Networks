import sys
from configparser import ConfigParser

from ResUNet_model import build_network
from utils import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))  # Change working dir
parser = ConfigParser()


def dice_coefficient(y_true, y_pred, n_labels):
    y_true = K.cast(y_true, "float")
    dice_coef = 0
    for i in range(n_labels):
        intersection = tf.reduce_sum(
            y_pred[:, :, :, :, i] * (y_true[:, :, :, :, i]))
        union_prediction = tf.reduce_sum(
            y_pred[:, :, :, :, i] * y_pred[:, :, :, :, i])
        union_ground_truth = tf.reduce_sum(
            (y_true[:, :, :, :, i]) * (y_true[:, :, :, :, i]))
        union = union_ground_truth + union_prediction
        dice_coef += (2 * intersection / union)
    return dice_coef / n_labels


def dice_loss(y_true, y_pred, n_labels):
    return 1.0 - dice_coefficient(y_true, y_pred, n_labels)


def build_data_generator(path, batch_size, sample_weight=None):
    files = glob(os.path.join(path, '*.npz'))
    print(' Number of files: ', np.shape(files))

    while True:
        shuffle(files)
        n_file = -1

        for npz in files:
            n_file += 1
            archive = np.load(npz)

            images = archive['t1_flair'] if 't1_flair' in archive \
                else archive['images']
            labels = archive['labels']
            del archive
            assert len(images) == 1024 or len(images) == 1

            for i in range(0, len(images), batch_size):
                end_i = min(i + batch_size, len(images))
                t1_flair_batch = images[i:end_i]
                labels_batch = labels[i:end_i]

                if sample_weight == None:
                    yield t1_flair_batch, labels_batch
                else:
                    yield t1_flair_batch, labels_batch, sample_weight


def train_unet(dtset_arch: str, model_n: int, p_selforth: float,
               p_interorth: float):
    initial_learning_rate = parser["TRAIN"].getfloat("learning_rate")
    lrd = parser["TRAIN"].getfloat("learning_rate_decay")
    batch_size = parser["TRAIN"].getint("batch_size")
    epochs = parser["TRAIN"].getint("epochs")
    input_channels = parser["TRAIN"].getint("input_channels")
    output_channels = parser["TRAIN"].getint("output_channels")
    model_no = 'model_{}'.format(model)

    f_train = open(os.path.join(patches_directory, "metadata_train.txt"), "r")
    n_train = int(f_train.read())
    f_train.close()

    steps_per_epoch = (n_train // 50 // batch_size)

    f_val = open(os.path.join(patches_directory, "metadata_val.txt"), "r")
    n_val = int(f_val.read())
    f_val.close()
    val_steps = (n_val // batch_size)

    print("-------- PARAMETERS")
    print("n_train = {}".format(n_train))
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("steps_per_epoch:", steps_per_epoch)
    print("Validation Steps:", val_steps)

    def div_regularization(kernel_name, model_number):
        training_kernel = unet.get_layer(kernel_name).weights[0]
        kernel = training_kernel
        [kh, kw, kd, i_c, o_c] = kernel.shape
        kernel = tf.reshape(kernel, (kh * kw * kd * i_c, o_c))
        kernel = K.transpose(kernel)
        kernel_norm = K.l2_normalize(kernel, axis=-1)
        sim_matrix = K.dot(kernel_norm, K.transpose(kernel_norm))
        diag_mask = (K.ones(sim_matrix.shape) - K.eye((sim_matrix).shape[0]))
        sim_matrix = sim_matrix * diag_mask  # removing the diagonal elements
        self_error = 0.5 * K.sum(K.square(K.abs(sim_matrix)))

        inter_models_error = 0
        if (ensemble == 'inter-orthogonal') and (model_number != 0):
            for n_model in range(model_number):
                reference = reference_weights[
                    n_model].get_layer(kernel_name).weights[0]
                reference = tf.reshape(reference, (kh * kw * kd * i_c, o_c))
                reference = K.transpose(reference)
                reference_norm = K.l2_normalize(reference, axis=-1)
                sim_matrix = K.dot(kernel_norm, K.transpose(reference_norm))
                model_error = K.sum(K.square(K.abs(sim_matrix)))
                inter_models_error += model_error

            inter_models_error = inter_models_error / model_number

        return self_error, inter_models_error

    def train_on_batch(xy):
        x, y = xy
        with tf.GradientTape() as tape:
            segmentation = unet(x)
            loss_reconstruction = dice_loss(y, segmentation, n_labels=1)

            self_orthogonal_loss, inter_orthogonal_loss = 0, 0
            if ensemble in ['inter-orthogonal', 'self-orthogonal']:
                for layer in range(1, 17):
                    self_o, inter_o = div_regularization(f'conv3d_{layer}',
                                                         model_n)
                    inter_orthogonal_loss += inter_o
                    self_orthogonal_loss += self_o

            final_loss = loss_reconstruction + \
                         p_selforth * self_orthogonal_loss + \
                         p_interorth * inter_orthogonal_loss

            gradients = tape.gradient(final_loss, unet.trainable_weights)

        optimizer.apply_gradients(zip(gradients, unet.trainable_weights))
        return loss_reconstruction, inter_orthogonal_loss, \
               self_orthogonal_loss, final_loss

    @tf.function
    def val_on_batch(xgt):
        x, gt = xgt
        segmentation = unet(x)
        loss = dice_loss(gt, segmentation, output_channels)
        dc = dice_coefficient(gt, segmentation, output_channels)
        return loss, dc

    train_dir = os.path.join(patches_directory, 'train')
    train_generator = build_data_generator(train_dir, batch_size)
    val_dir = os.path.join(patches_directory, 'val')
    val_generator = build_data_generator(val_dir, batch_size)

    inputs, outputs = build_network(input_channels, output_channels)
    unet = keras.Model(inputs, outputs)
    unet.summary()
    optimizer = tf.keras.optimizers.Adam(lr=initial_learning_rate)

    summary_writer = tf.summary.create_file_writer(
        os.path.join(logs_directory, dtset_arch, model_no))

    model_folder = os.path.join(models_directory, dtset_arch)
    os.makedirs(model_folder, exist_ok=True)
    print(f"Model will be saved in: {model_folder}.")

    if ensemble == 'inter-orthogonal':
        reference_weights = list()
        for i in range(model_n):
            model_ref_name = 'model_' + str(i)
            model_ref = load_model(
                os.path.join(models_directory, dtset_arch, model_ref_name))
            reference_weights.append(model_ref)

    best_loss_val_value = None
    best_loss_val_epoch = 0

    for epoch in range(1, epochs):
        print('--- Epoch  ', epoch, ' /', epochs)
        cum_inter_orth = list()
        cum_self_orth = list()
        cum_rec = list()
        cum_loss = list()

        for step in range(steps_per_epoch):
            data = next(train_generator)
            loss_rec, inter_orth, self_orth, final_loss = train_on_batch(data)
            cum_loss.append(final_loss)
            cum_self_orth.append(self_orth)
            cum_inter_orth.append(inter_orth)
            cum_rec.append(loss_rec)

        if (epoch - 1) % 1 == 0:
            val_loss = list()
            mean_dice = list()
            for vfiles in range(val_steps):
                val_data = next(val_generator)
                loss, dc = val_on_batch(val_data)
                val_loss.append(loss)
                mean_dice.append(dc)
            avg_val_loss = np.mean(val_loss)
            if epoch == 1 or (epoch > 1 and
                              avg_val_loss < best_loss_val_value):
                best_loss_val_epoch = epoch
                best_loss_val_value = avg_val_loss

                print(f'New best model found. Val loss {best_loss_val_value})')
                model_path = os.path.join(models_directory, dtset_arch,
                                          model_no)
                print(f'Saving model: {model_path}. If it already existed, it '
                      f'was overwrited.')
                save_model(unet, model_path)

            with summary_writer.as_default():
                tf.summary.scalar('trainining_loss', np.mean(cum_loss),
                                  step=epoch)
                tf.summary.scalar('training_dice-loss', np.mean(cum_rec),
                                  step=epoch)
                tf.summary.scalar('trainig_inter-orthogonal_loss',
                                  np.mean(cum_inter_orth), step=epoch)
                tf.summary.scalar('training_self-orthogonal_loss',
                                  np.mean(cum_self_orth), step=epoch)
                tf.summary.scalar('validation-loss', np.mean(val_loss),
                                  step=epoch)
                tf.summary.scalar('validation-dice: ', np.mean(mean_dice),
                                  step=epoch)
                tf.summary.scalar('Learning_rate: ',
                                  optimizer.learning_rate.numpy(), step=epoch)

            print('Epoch: ', epoch, '  Validation dice coeff: ',
                  np.mean(mean_dice))

        if epoch % 10 == 0:
            optimizer.learning_rate.assign(optimizer.learning_rate * lrd)

    print(f'Best model: epoch {best_loss_val_epoch}, '
          f'loss {best_loss_val_value}')

    del unet
    del optimizer
    summary_writer.close()
    K.clear_session()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError(f"When running 'python training_ensemble.py' you "
                             f"must provide the path of the configuration file"
                             f" (.ini).")
    ini_file = sys.argv[1]
    parser.read(ini_file)

    dataset_name = parser["ENSEMBLE"].get("dataset")
    patches_directory = parser["DEFAULT"].get("patches_directory")
    models_directory = parser["DEFAULT"].get("models_directory")
    logs_directory = parser["DEFAULT"].get("logs_directory")

    if os.path.isdir(patches_directory):
        print('patches directory: ', patches_directory)
    else:
        raise FileNotFoundError(f' ------ patches directory does not exist ('
                                f'{patches_directory}).')

    ensemble = parser["ENSEMBLE"].get("ensemble")
    n_models = parser["ENSEMBLE"].getint("n_models")
    network = parser["ENSEMBLE"].get("network")

    model_fold = dataset_name + '_' + network
    n, self_p, inter_p = None, 0, 0
    if ensemble == 'random':  # TODO No le pasa el model_n en el original
        model_fold += '_random'
    elif ensemble == 'self-orthogonal':
        self_p = parser["ENSEMBLE"].getfloat("self_p")
        model_fold += f'_self-orthogonal_selfp_{self_p}'
    elif ensemble == 'inter-orthogonal':
        self_p = parser["ENSEMBLE"].getfloat("self_p")
        inter_p = parser["ENSEMBLE"].getfloat("inter_p")
        model_fold += f'_inter-orthogonal_selfp_{self_p}_interp_{inter_p}'
    else:
        raise AttributeError(f'The ensemble type ({ensemble}) is not valid.')

    for model in range(n_models):
        print('model: ', model)
        train_unet(model_fold, model, self_p, inter_p)

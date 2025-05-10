import tensorflow as tf
from tqdm import tqdm
import IPython.display as ipd
import pickle
import os

from ..data.dataset import create_dataset, prepare_datasets
from ..data.labels import process_labels, get_stratified_sample
from .losses import generator_loss, discriminator_loss
from .generator import Generator
from .discriminator import Discriminator

class GAN:
    def __init__(self, rgb_image_path, ir_image_path, label_file_path):
        self.generator = Generator()
        self.discriminator = Discriminator()
        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        # Process labels and create datasets
        self.labs = process_labels(label_file_path)
        self.stratified_sample = get_stratified_sample(self.labs)
        
        # Create datasets
        self.dataset = create_dataset(rgb_image_path, ir_image_path, list(self.labs.keys()), allData=True)
        self.TOTAL_IMAGES = len(self.dataset)
        self.train_size = int(0.7 * self.TOTAL_IMAGES)
        self.val_size = int(0.2 * self.TOTAL_IMAGES)
        
        self.train_ds, self.val_ds, self.test_ds = prepare_datasets(
            self.dataset, self.train_size, self.val_size
        )

    @tf.function
    def train_step(self, inputs, target):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_output = self.generator(inputs, training=True)

            discriminator_real_output = self.discriminator([inputs, target], training=True)
            discriminator_generated_output = self.discriminator([inputs, generated_output], training=True)

            generator_total_loss, generator_gan_loss, generator_l1_loss = generator_loss(
                discriminator_generated_output, generated_output, target
            )

            discriminator_loss_val = discriminator_loss(
                discriminator_real_output, discriminator_generated_output
            )

        generator_gradients = generator_tape.gradient(
            generator_total_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss_val, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        return generator_total_loss, discriminator_loss_val

    def fit(self, epochs, patience=None, checkpoint_dir=None):
        val_gen_loss, val_disc_loss = [], []
        train_gen_loss, train_disc_loss = [], []
        best_val_gen_loss = float('inf')
        best_val_disc_loss = float('inf')
        wait = 0

        for epoch in tqdm(range(epochs)):
            for rgb, ir in tqdm(self.train_ds, leave=False):
                gen_loss, disc_loss = self.train_step(rgb, ir)

            ipd.clear_output(wait=True)
            print(f"Training for epoch {epoch+1} done.")

            # Validation
            val_generator_loss, val_discriminator_loss = 0, 0
            for val_rgb, val_ir in self.val_ds:
                val_gen_output = self.generator(val_rgb, training=False)
                val_disc_real = self.discriminator([val_rgb, val_ir], training=False)
                val_disc_fake = self.discriminator([val_rgb, val_gen_output], training=False)
                
                gen_loss, _, _ = generator_loss(val_disc_fake, val_gen_output, val_ir)
                disc_loss = discriminator_loss(val_disc_real, val_disc_fake)

                val_generator_loss += gen_loss
                val_discriminator_loss += disc_loss

            train_gen_loss.append(gen_loss)
            train_disc_loss.append(disc_loss)
            val_generator_loss /= len(self.val_ds)
            val_discriminator_loss /= len(self.val_ds)
            val_gen_loss.append(val_generator_loss)
            val_disc_loss.append(val_discriminator_loss)

            print(f"Train Generator Loss = {gen_loss:.4f}")
            print(f"Train Discriminator Loss = {disc_loss:.4f}")
            print(f"Val. Generator Loss = {val_generator_loss:.4f}")
            print(f"Val. Discriminator Loss = {val_discriminator_loss:.4f}\n")

            # Early stopping
            if patience:
                if val_generator_loss < best_val_gen_loss:
                    best_val_gen_loss = val_generator_loss
                    wait = 0
                    if checkpoint_dir:
                        self.save_models(checkpoint_dir)
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping triggered")
                        break

        history = {
            'train_gen_loss': train_gen_loss,
            'train_disc_loss': train_disc_loss,
            'val_gen_loss': val_gen_loss,
            'val_disc_loss': val_disc_loss
        }
        
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'training_history.pkl'), 'wb') as f:
                pickle.dump(history, f)
        
        return history

    def save_models(self, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.generator.save(os.path.join(checkpoint_dir, 'generator.keras'))
        self.discriminator.save(os.path.join(checkpoint_dir, 'discriminator.keras')) 

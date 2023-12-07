import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False  # To prevent it from being trained again during style transfer

def load_img(path_to_img):
    img = tf.keras.preprocessing.image.load_img(path_to_img, target_size=(img_nrows, img_ncols))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Convert from BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_feature_representations(model, content_img, style_img):
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    content_outputs = [model.get_layer(layer).output for layer in content_layers]
    style_outputs = [model.get_layer(layer).output for layer in style_layers]

    model_outputs = content_outputs + style_outputs
    model = tf.keras.models.Model(inputs=model.input, outputs=model_outputs)

    content_features = model(content_img)
    style_features = model(style_img)

    return content_features, style_features

def content_loss(content_features, generated_features):
    content_losses = []
    for content_feat, gen_feat in zip(content_features, generated_features):
        current_content_loss = tf.reduce_mean(tf.square(content_feat - gen_feat))
        content_losses.append(current_content_loss)
    return tf.reduce_mean(content_losses)

def gram_matrix(input_tensor):
    # Calculate Gram matrix for style loss
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(style_features, generated_features):
    style_losses = []
    for style_feat, gen_feat in zip(style_features, generated_features):
        style_gram = gram_matrix(style_feat)
        gen_gram = gram_matrix(gen_feat)
        layer_style_loss = tf.reduce_mean(tf.square(style_gram - gen_gram))
        style_losses.append(layer_style_loss)
    return tf.reduce_mean(style_losses)


# Set up paths to content and style images
content_path = '/Users/elliottbarnes/Documents/Dev/GitHub/ML/neural_style_transfer/dark_castle_by_emkun_d84t11d-fullview.jpg'
style_path = '/Users/elliottbarnes/documents/dev/github/ml/neural_style_transfer/Van_Gogh_style_photo_effect.jpeg'

# Set dimensions for the images
img_nrows = 400  # Define the desired height for the images
width, height = tf.keras.preprocessing.image.load_img(content_path).size
img_ncols = int(width * img_nrows / height)

# Load the pre-trained VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Load and preprocess content and style images
content_img = load_img(content_path)
style_img = load_img(style_path)

# Extract content and style features
content_features, style_features = get_feature_representations(vgg, content_img, style_img)

# Set hyperparameters and optimization settings
epochs = 1000
style_weight = 1e-2
display_interval = 100

# Initialize the generated image
generated_img = tf.Variable(content_img, dtype=tf.float32)


# Optimization loop
optimizer = tf.optimizers.legacy.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

for i in range(epochs):
    with tf.GradientTape() as tape:
        content_feat_gen, style_feat_gen = get_feature_representations(vgg, generated_img, generated_img)
        c_loss = content_loss(content_features, content_feat_gen)
        s_loss = style_loss(style_features, style_feat_gen)
        total_loss = c_loss + (style_weight * s_loss)
    
    grads = tape.gradient(total_loss, generated_img)
    optimizer.apply_gradients([(grads, generated_img)])

    # Clip generated image values to stay within the valid range
    generated_img.assign(tf.clip_by_value(generated_img, 0.0, 255.0))
    gen_img = deprocess_img(generated_img.numpy())
    plt.imshow(gen_img)
    plt.title(f'Iteration: {i}')
    plt.show()
    time.sleep(0.5)

# Post-processing and display the final stylized image
final_img = deprocess_img(generated_img.numpy())
plt.imshow(final_img)
plt.title('Final Stylized Image')
plt.show()

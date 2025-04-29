# Function to pad text to block size
def pad_text(text, block_size):
    return text + (block_size - len(text) % block_size) * ' '

# DES Encryption
def des_encrypt(plain_text, key):
    cipher = DES.new(key, DES.MODE_ECB)
    padded_text = pad_text(plain_text, DES.block_size)
    encrypted_text = cipher.encrypt(padded_text.encode())
    return base64.b64encode(encrypted_text).decode()

# AES Encryption
def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_text = pad_text(plain_text, AES.block_size)
    encrypted_text = cipher.encrypt(padded_text.encode())
    return base64.b64encode(encrypted_text).decode()


# Image Loading
def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)
    return img, img_data

# Image Saving
def save_image(image_data, output_path):
    img = Image.fromarray(image_data)
    img.save(output_path)

# AES Image Encryption
def encrypt_image_aes(image_data, key):
    cipher = AES.new(key, AES.MODE_ECB)
    shape = image_data.shape
    flat_data = image_data.flatten()
    padded_data = pad(flat_data.tobytes(), AES.block_size)
    encrypted_data = cipher.encrypt(padded_data)
    encrypted_array = np.frombuffer(encrypted_data, dtype=np.uint8)[:flat_data.size]
    return encrypted_array.reshape(shape)

# Main Function
def main():
    plain_text = "HelloRoopam"
    des_key = b'SecretKe'  # 8 bytes key for DES
    aes_key = b'16bytekeylengthp'  # 16 bytes key for AES

    # DES Encryption Output
    des_cipher_text = des_encrypt(plain_text, des_key)
    print(f"DES Plain Text: {plain_text}")
    print(f"DES Cipher Text: {des_cipher_text}")
    print("Is Encryption Good? NO")

    # AES Encryption Output
    aes_cipher_text = aes_encrypt(plain_text, aes_key)
    print(f"AES Plain Text: {plain_text}")
    print(f"AES Cipher Text: {aes_cipher_text}")
    print("Is Encryption Good? YES")

    # Image Encryption
    input_image = 'roopam.png'  # Replace with your image path
    output_image = 'encrypted_image.png'

    img, img_data = load_image(input_image)
    encrypted_data = encrypt_image_aes(img_data, aes_key)
    save_image(encrypted_data, output_image)
    print(f"Encrypted image saved as {output_image}")

    # Display Original and Encrypted Images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    img_encrypted = Image.open(output_image)
    plt.imshow(img_encrypted, cmap='gray')
    plt.title("Encrypted Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()

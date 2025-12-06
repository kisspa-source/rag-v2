import hashlib
import secrets
import sys

def generate_hash(password: str, salt: str = None) -> tuple[str, str]:
    """
    Generate SHA-256 hash for a password with salt.
    If salt is not provided, generate a random one.
    """
    if not salt:
        salt = secrets.token_hex(16)
    
    # Combine password and salt
    salted_password = password + salt
    
    # Generate hash
    password_hash = hashlib.sha256(salted_password.encode()).hexdigest()
    
    return password_hash, salt

if __name__ == "__main__":
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        password = input("Enter password to hash: ")
    
    if not password:
        print("Password cannot be empty.")
        sys.exit(1)
        
    p_hash, p_salt = generate_hash(password)
    
    print(f"\nPassword: {password}")
    print(f"Hash: {p_hash}")
    print(f"Salt: {p_salt}")
    print("\nAdd these to your config/config.yaml under 'security':")
    print(f"  admin_password_hash: \"{p_hash}\"")
    print(f"  salt: \"{p_salt}\"")

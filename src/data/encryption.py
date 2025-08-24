import os
import stat
import getpass
import paramiko
import pandas as pd
from io import BytesIO
from pathlib import Path
from cryptography.fernet import Fernet


def generate_and_secure_key_in_local_env_file(
    key_var_name: str,
    env_file_path: str = ".env",
    force_overwrite: bool = False,
) -> None:
    """
    Save an encryption key to an environment file with owner-only restricted permissions

    Args:
        key_var_name (str): name for the environment variable
        force_overwrite (bool): if True, overwrites an existing .env file without asking
    """
    # Prevent accidental overwriting of an existing environment file
    if Path(env_file_path).exists() and not force_overwrite:
        confirm = input(
            f"WARNING: The file '{env_file_path}' already exists. "
            "Overwrite it? You may lose access to previously encrypted data. (y/n): "
        ).lower()
        if confirm not in ('y', 'yes'):
            print("Operation cancelled. Your existing environment file was not modified.")
            return

    # Write the key to the environment file
    print("Generating a new encryption key...")
    key = Fernet.generate_key().decode()
    try:
        with open(env_file_path, "w") as f:
            f.write(f'{key_var_name}="{key}"\n')
        print(f"Successfully wrote key to '{env_file_path}'.")

    except IOError as e:
        print(f"Error: Could not write to file '{env_file_path}'. Reason: {e}")
        return

    # Set secure file permissions (for Linux, macOS, etc.)
    print(f"Setting permissions set to owner-only for '{env_file_path}'.")
    if os.name == "posix":
        os.chmod(env_file_path, stat.S_IRUSR | stat.S_IWUSR)
    else:
        print(
            "Warning: Setting file permissions is not supported on this OS. "
            "Please ensure the file is secured manually."
        )


def get_key(encryption_key_var_name: str) -> bytes:
    """
    Retrieve the encryption key from an environment variable

    Raises:
        ValueError: If the environment variable is not set.
    """
    key_str = os.getenv(encryption_key_var_name)
    if not key_str:
        raise ValueError(
            f"'{encryption_key_var_name}' environment variable not set! "
            "Please generate a key and set it."
        )
    return key_str.encode()


def write_pandas_to_encrypted_file(
    data_to_encrypt: pd.DataFrame,
    encrypted_file_path: str,
    encryption_key_var_name: str,
) -> None:
    """
    Encrypts a DataFrame and saves it to a binary file

    Args:
        data_to_encrypt: pandas DataFrame data to encrypt
        encrypted_file_path: full path where the encrypted file will be saved
        encryption_key_var_name: name of the environment variable holding the key.
    """
    key = get_key(encryption_key_var_name)
    f = Fernet(key)

    # Ensure the output directory exists
    output_dir = os.path.dirname(encrypted_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert DataFrame to CSV bytes in memory
    csv_bytes = data_to_encrypt.to_csv(index=False).encode('utf-8')
    encrypted_data = f.encrypt(csv_bytes)

    # Write the encrypted bytes to the specified file
    with open(encrypted_file_path, "wb") as encrypted_file:
        encrypted_file.write(encrypted_data)

    print(f"Data encrypted and saved to {encrypted_file_path}")


def read_pandas_from_encrypted_file(
    encrypted_file_path: str,
    encryption_key_var_name: str,
) -> pd.DataFrame:
    """
    Reads and decrypts a file into a pandas DataFrame.

    Args:
        encrypted_file_path: path to the encrypted file.
        encryption_key_var_name: name of the environment variable holding the key

    Returns:
        A pandas DataFrame with the decrypted data.
    """
    key = get_key(encryption_key_var_name)
    f = Fernet(key)

    with open(encrypted_file_path, "rb") as encrypted_file:
        encrypted_data = encrypted_file.read()

    decrypted_data = f.decrypt(encrypted_data)
    df = pd.read_csv(BytesIO(decrypted_data))

    return df


def load_remote_dotenv(
    hostname: str,
    username: str,
    remote_env_path: str,
    port: int = 22,
    password: str = None,
    private_key_path: str = None,
) -> bool:
    """
    Connects to a remote server via SSH, reads a .env file, and loads its
    variables into the current environment.

    Args:
        hostname (str): server's hostname or IP address
        username (str): username for the SSH connection
        remote_env_path (str): full path to the .env file on the remote server
        port (int): SSH port (default is 22)
        private_key_path (str, optional): path to your private SSH key for auth

    Returns:
        bool: success of the environment variables loading operation
    
    Example:
        success = load_remote_dotenv(
            hostname="example.com",
            username="user",
            remote_env_path="/home/user/.env",
            private_key_path="/path/to/private/key"
        )
    """
    # Prepare authentication for client connection
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = {
        "hostname": hostname,
        "port": port,
        "username": username,
    }

    try:

        # Prioritize key-based authentication if a path is provided
        if private_key_path:
            private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
            connect_kwargs["pkey"] = private_key
            print("Attempting SSH connection with private key...")

        # If no key, prompt for password
        else:
            if password is None:
                password = getpass.getpass(f"Enter password for {username}@{hostname}: ")
            connect_kwargs["password"] = password
            print("Attempting SSH connection with password...")
            
        client.connect(**connect_kwargs)

        print("Connection successful. Opening SFTP session...")
        sftp_client = client.open_sftp()

        print(f"Reading remote environment file: {remote_env_path}")
        with sftp_client.open(remote_env_path, "r") as remote_file:
            for line in remote_file:
                line: str
                line = line.strip()
                # Ignore comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Split on the first "="
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')  # cleaning up quotes/whitespace
                    
                    # Set the environment variable for the current process
                    os.environ[key] = value
                    print(f"Loaded environment variable: '{key}'")
        
        print("Successfully loaded all variables from remote .env file.")
        return True

    # Exceptions for SSH connection issues, and final cleanup
    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your credentials or SSH key.")
        return False
    except paramiko.SSHException as e:
        print(f"Unable to establish SSH connection: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Remote file not found at '{remote_env_path}' on {hostname}.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    finally:
        if client:
            client.close()
        print("Connection closed.")

import jax


def default_device() -> jax.Device:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    return devices[0] if devices else jax.devices("cpu")[0]

    

def resolve_device(device: str | jax.Device | None) -> jax.Device | None:
    if device is None:
        return None

    if isinstance(device, str):
        if device == "cuda":
            device = "gpu"

        if ":" in device:
            device_type, index = device.split(":")
            if device_type == "cuda":
                device_type = "gpu"
            return jax.devices(device_type)[int(index)]

        return jax.devices(device)[0]

    return device
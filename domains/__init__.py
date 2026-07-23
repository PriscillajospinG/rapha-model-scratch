from . import lower_limb, upper_body

_REGISTRY = {
    lower_limb.CONFIG.name: lower_limb.CONFIG,
    upper_body.CONFIG.name: upper_body.CONFIG,
}

DOMAIN_NAMES = list(_REGISTRY.keys())


def get_domain(name):
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown domain '{name}'. Available: {DOMAIN_NAMES}. "
            f"(face and stroke_rehab are not standalone landmark domains -- "
            f"see README.md for their status.)"
        )
    return _REGISTRY[name]


import cv2


def apply_ycrcb_clahe(
    image,
    clip_limit=1.8,
    tile_size=16,
    alpha=0.6,
    bilateral_d=5,
    sigma_color=35,
    sigma_space=35
):
    """
    Apply noise-aware CLAHE to the luminance channel
    while preserving the original chrominance channels.
    """

    if image is None or len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected a valid 3-channel BGR image")

    if clip_limit <= 0:
        raise ValueError("Clip limit must be greater than zero")

    if tile_size < 1:
        raise ValueError("Tile size must be at least 1")

    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    if bilateral_d < 1:
        raise ValueError("Bilateral filter diameter must be positive")

    # Convert BGR to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Step 1: Denoise luminance while preserving edges
    denoised_y = cv2.bilateralFilter(
        y,
        d=bilateral_d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    # Step 2: Apply CLAHE with controlled parameters
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    enhanced_y = clahe.apply(denoised_y)

    # Step 3: Blend CLAHE output with original luminance
    blended_y = cv2.addWeighted(
        enhanced_y,
        alpha,
        y,
        1.0 - alpha,
        0
    )

    # Step 4: Recombine with original chrominance
    enhanced_ycrcb = cv2.merge((
        blended_y,
        cr,
        cb
    ))

    # Convert back to BGR
    enhanced_image = cv2.cvtColor(
        enhanced_ycrcb,
        cv2.COLOR_YCrCb2BGR
    )

    return enhanced_image

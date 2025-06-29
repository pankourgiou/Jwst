import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, LogStretch
from scipy.ndimage import gaussian_filter, rotate

# === Parameters ===
size = 1024  # Image size (pixels)
num_stars = 200
num_galaxies = 50

# === Create a dark infrared background ===
np.random.seed(42)
image = np.random.normal(loc=800, scale=20, size=(size, size))  # cosmic background

# === Add stars ===
for _ in range(num_stars):
    x, y = np.random.randint(10, size-10, 2)
    brightness = np.random.uniform(300, 3000)
    star = np.zeros((size, size))
    star[y, x] = brightness
    star = gaussian_filter(star, sigma=np.random.uniform(0.8, 2.5))
    image += star

# === Add galaxies ===
for _ in range(num_galaxies):
    x, y = np.random.randint(50, size-50, 2)
    major = np.random.uniform(5, 20)
    minor = major * np.random.uniform(0.3, 1.0)
    angle = np.random.uniform(0, 180)
    brightness = np.random.uniform(500, 5000)
    
    galaxy = np.zeros((50, 50))
    cy, cx = 25, 25
    yy, xx = np.ogrid[:50, :50]
    ellipse = ((xx - cx)**2 / major**2) + ((yy - cy)**2 / minor**2)
    galaxy[ellipse <= 1] = brightness
    galaxy = gaussian_filter(galaxy, sigma=2.5)
    galaxy = rotate(galaxy, angle, reshape=False)
    
    image[y-25:y+25, x-25:x+25] += galaxy

# === Add detector artifacts ===
gradient = np.linspace(0.95, 1.05, size)
image = image * gradient[:, None]  # vertical vignetting
image += np.random.normal(scale=10, size=(size, size))  # read noise

# === Final cleanup ===
image = np.clip(image, 0, None)

# === Save to FITS with metadata ===
hdu = fits.PrimaryHDU(image)
hdu.header['TELESCOP'] = 'JWST-SIM'
hdu.header['INSTRUME'] = 'NIRCam'
hdu.header['OBSERVER'] = 'OpenAI-Sim'
hdu.header['COMMENT'] = "Simulated JWST deep field image with stars and galaxies"
fits.writeto("jwst_deepfield_sim.fits", hdu.data, hdu.header, overwrite=True)

# === Display ===
norm = ImageNormalize(image, stretch=LogStretch())

plt.figure(figsize=(10, 8))
plt.imshow(image, cmap='inferno', origin='lower', norm=norm)
plt.title("Simulated JWST Deep Field (Stars + Galaxies)")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.colorbar(label='Flux (arbitrary units)')
plt.tight_layout()
plt.show()

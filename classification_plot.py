import numpy as np
from scipy.ndimage import label
from matplotlib.colors import ListedColormap
from crism_ml.preprocessing import label_to_index
from crism_ml.train import _to_coords


mineral_colours = {
    "Saponite": [0, 118 / 256, 118 / 256],
    "Jarosite": [56 / 256, 135 / 256, 70 / 256],
    "Nontronite": [57 / 256, 198 / 256, 187 / 256],
    "MgCO3": [22 / 256, 222 / 256, 233 / 256],
    "Ca/Fe CO3": [22 / 256, 116 / 256, 233 / 256],
    "Halloysite": [104 / 256, 122 / 256, 68 / 256],
    "Kaolinite": [132 / 256, 155 / 256, 86 / 256],
    "Illite/Muscovite": [206 / 256, 214 / 256, 41 / 256],
    "Alunite": [175 / 256, 169 / 256, 80 / 256],
    "Akaganeite": [94 / 256, 160 / 256, 161 / 256],
    "Serpentine": [114 / 256, 25 / 256, 230 / 256],
    "Chlorite": [23 / 256, 232 / 256, 54 / 256],
    "Clinochlore": [199 / 256, 230 / 256, 23 / 256],
    "Monohydrated sulfate": [229 / 256, 26 / 256, 223 / 256],
    "Polyhydrated sulfate": [229 / 256, 26 / 256, 122 / 256],
    "Prehnite": [256 / 256, 179 / 256, 0 / 256],
    "Opal 1": [184 / 256, 90 / 256, 71 / 256],
    "Opal 2": [184 / 256, 71 / 256, 165 / 256],
    "Chloride": [211 / 256, 159 / 256, 44 / 256],
    "Gypsum": [137 / 256, 256 / 256, 0 / 256],
    "Low Ca Pyroxene": [106 / 256, 57 / 256, 155 / 256],
    "High Ca Pyroxene": [155 / 256, 57 / 256, 155 / 256],
    "Olivine Fayalite": [188 / 256, 19 / 256, 61 / 256],
    "Olivine Forsterite": [188 / 256, 62 / 256, 19 / 256],
}


CLASS_NAMES = {
    0: "CO2 Ice",  # co2_ice
    1: "H2O Ice",  # h20_ice
    2: "Gypsum",  # gypsum
    3: "Ferric Hydroxysulfate",  # hydrox_fe_sulf
    4: "Hematite",  # hematite
    5: "Nontronite",  # fe_smectite
    6: "Saponite",  # mg_smectite
    7: "Prehnite",  # Prehnite Zeolite # prehnite
    8: "Jarosite",  # jarosite
    9: "Serpentine",  # serpentine
    10: "Alunite",  # alunite
    11: "Akaganeite",  # Fe Oxyhydroxysulfate # hydrox_fe_sulf
    12: "Ca/Fe CO3",  # Calcite, Ca/Fe carbonate  # fe_ca_carbonate
    13: "Beidellite",  # Al-smectite # al_smectite
    14: "Kaolinite",  # kaolinite
    15: "Bassanite",  # bassanite
    16: "Epidote",  # epidote
    17: "Montmorillonite",  # Al-smectite # al_smectite
    18: "Rosenite",  # Polyhydrated sulfate # poly_hyd_sulf
    19: "Mg Cl salt",  # Mg(ClO3)2.6H2O # Polyhydrated sulfate # poly_hyd_sulf
    20: "Halloysite",  # Kaolinite # kaolinite
    21: "Bland",  # Neutral/no spectral features
    22: "Illite/Muscovite",  # illite_muscovite
    23: "Margarite",  # Illite/Muscovite # illite_muscovite
    24: "Analcime",  # Zeolite # analcime
    25: "Monohydrated sulfate",  # Szomolnokite # mono_hyd_sulf
    26: "Opal 1",  # Opal # Hydrated silica # hydrated_silica
    27: "Opal 2",  # Opal-A # Hydrated silica # hydrated_silica
    28: "Iron Oxide Silicate Sulfate",  # Polyhydrated sulfate # poly_hyd_sulf
    29: "MgCO3",  # Magnesite # mg_carbonate
    30: "Chlorite",  # chlorite
    31: "Clinochlore",  # chlorite
    32: "Low Ca Pyroxene",  # lcp
    33: "Olivine Forsterite",  # mg_olivine
    34: "High Ca Pyroxene",  # hcp
    35: "Olivine Fayalite",  # fe_olivine
    36: "Chloride",  # chloride
    37: "Artefact",  # Camera artefact described in Leask et al. 2018
}


def preds_to_coords(preds: np.ndarray, min_size: int = 0) -> np.ndarray:
    """Convert a 2D array of predictions to a list of coordinates.
    Optionally filter by minimum size of connected component.

    Parameters
    ----------
    preds : np.ndarray
        2D array of predictions.
    min_size : int
        Minimum size of connected component to keep.
        Default is 0.

    Returns
    -------
    coords : np.ndarray
        Array of coordinates.
    """
    pred_labels, n_obj = label(preds, structure=np.ones((3, 3)))
    total_indices = [
        area
        for area in label_to_index(pred_labels, n_obj)[1:]
        if area.size >= min_size
    ]
    coords = [
        _to_coords(indices, pred_labels.shape) for indices in total_indices
    ]
    if len(coords) == 0:
        return None
    coords = np.concatenate(coords)
    return coords


def convert_to_coords_filter_regions_by_conf(
    image_pred: np.ndarray,
    confidence_scores: np.ndarray,
    min_area: int = 0,
    min_confidence: float = 0.0,
) -> np.ndarray:
    """Convert mineral prediction to dictionary of per-class coordinates.
    Optionally filter by minimum area of connected components of a single class
    and minimum model confidence score, averaged across the connected
    component.

    Parameters
    ----------
    image_pred : np.ndarray
        2D array of predictions.
    confidence_scores : np.ndarray
        2D array of confidence scores.
    min_area : int
        Minimum area of connected component to keep.
        Default is 0.
    min_confidence : float, 0 to 1.
        Minimum confidence score to keep.
        Default is 0.0.

    Returns
    -------
    coord_dict : dict
        Dictionary of coordinates for each class.
        Coordinates given as [[x, y], n_pixels].
    """
    unique_preds = np.unique(image_pred)
    coord_dict = {}
    for pred_class in unique_preds:
        class_mask = (image_pred == pred_class)
        # Get connected components
        class_pred_labels, n_obj = label(
            class_mask, structure=np.ones((3, 3))
        )
        # Filter connected components by size
        total_indices = [
            area
            for area in label_to_index(class_pred_labels, n_obj)[1:]
            if area.size >= min_area
        ]
        # total_indices is a list of indices for each connected component
        class_coords = []
        for region in total_indices:  # Loop through each connected component
            # Convert indices to x,y coordinates
            coords = np.flip(
                np.stack(np.unravel_index(region, image_pred.shape)), axis=0
            )
            # Calculate average confidence of region
            region_confidence = np.average(
                confidence_scores[coords[1], coords[0]]
            )
            if region_confidence > min_confidence:  # Filter by confidence
                class_coords.append(coords)
        # Concatenate the regions together
        if len(class_coords) > 0:
            class_coords = np.concatenate(
                class_coords, axis=1
            )
        coord_dict[pred_class] = np.array(class_coords)
    return coord_dict


def create_cmap(mineral_color: list, static: bool = False) -> ListedColormap:
    """Create a matplotlib colormap from white to the specified colour.
    Can also create a colormap of just the specified colour.

    Parameters
    ----------
    mineral_color : list
        List of RGB values.
    static : bool
        If True, create a static colormap of just the specified colour.
        Default is False.
    Returns
    -------
    cmap : ListedColormap
        Colormap.
    """
    if static:
        cmap_vals = np.ones((256, 4))
        cmap_vals[:, :3] = mineral_color
        cmap = ListedColormap(cmap_vals)
        return cmap
    else:
        cmap_vals = np.ones((256, 4))
        for i, val in enumerate(mineral_color):
            cmap_vals[:, i] = np.linspace(1, val, 256)
        cmap = ListedColormap(cmap_vals)
        return cmap


def get_mineral_conf_score_mask(
    pred_coords: dict, confidence_scores: np.ndarray, mineral: int
) -> np.ndarray:
    """Create a mask of the confidence scores for a specific mineral.

    Parameters
    ----------
    pred_coords : dict
        Dictionary of coordinates for each mineral.
    confidence_scores : np.ndarray
        Array of confidence scores.
    mineral : int
        Mineral label.

    Returns
    -------
    mask : np.ndarray
        Mask of confidence scores.
    """
    mask = np.zeros(confidence_scores.shape)
    coords = pred_coords[mineral]
    mask[coords[1], coords[0]] = confidence_scores[coords[1], coords[0]]
    return mask


def get_mean_mineral_spectra(
    image: np.ndarray, coords: np.ndarray
) -> np.ndarray:
    """Get the average spectra for a given mineral in an image.

    Parameters
    ----------
    image : np.ndarray
        Image cube of shape (n_rows, n_columns, n_bands)
    coords : np.ndarray
        Array of coordinates given as [[x, y], n_pixels]
    Returns
    -------
    avg_spectra : np.ndarray
        Array of mean spectra.
    """
    spectra = image[coords[1], coords[0]]
    spectra = spectra.reshape(-1, image.shape[-1])
    avg_spectra = np.average(spectra, axis=0)
    return avg_spectra

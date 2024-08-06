from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
from config import Segmentationconfig

import math
from PIL import Image


@dataclass
class TissueSegment:
    contour: np.ndarray
    holes: List[np.ndarray]


class TissueSegmenter:
    # Inspired from https://github.com/mahmoodlab/CLAM/blob/master/wsi_core/WholeSlideImage.py

    def __init__(self, config: Segmentationconfig):
        self.config = config
        self.slide_image = None

    def segment(self, wsi) -> List[TissueSegment]:
        self.slide_image = wsi
        img = np.array(wsi.wsi.read_region((0, 0), self.config.seg_level, wsi.wsi.level_dimensions[self.config.seg_level]))

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], self.config.mthresh)

        if self.config.use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, self.config.sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, self.config.sthresh, self.config.sthresh_up, cv2.THRESH_BINARY)

        if self.config.close > 0:
            kernel = np.ones((self.config.close, self.config.close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        scale = wsi.level_downsamples[self.config.seg_level]
        scaled_ref_patch_area = int(self.config.ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = self.config.filter_params.copy()
        filter_params["a_t"] = filter_params["a_t"] * scaled_ref_patch_area
        filter_params["a_h"] = filter_params["a_h"] * scaled_ref_patch_area

        foreground_contours, hole_contours = self._filter_contours(contours, hierarchy, filter_params)

        foreground_contours = self.scaleContourDim(foreground_contours, scale)
        hole_contours = self.scaleHolesDim(hole_contours, scale)

        foreground_contours, hole_contours = self._filter_thin_contours(foreground_contours, hole_contours)

        return [TissueSegment(contour, holes) for contour, holes in zip(foreground_contours, hole_contours)]

    def _filter_contours(self, contours, hierarchy, filter_params):
        """
        Filter contours by: area.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.array(hole_areas).sum()
            if a == 0:
                continue
            if tuple((filter_params["a_t"],)) < tuple((a,)):
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            # take max_n_holes largest holes by area
            unfilered_holes = unfilered_holes[: filter_params["max_n_holes"]]
            filtered_holes = []

            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > filter_params["a_h"]:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    def _filter_thin_contours(self, contours_tissue, holes_tissue):
        min_area = self.config.min_area
        min_aspect_ratio = self.config.min_aspect_ratio

        filtered_contours = []
        filtered_holes = []

        areas = []

        for contour, holes in zip(contours_tissue, holes_tissue):
            rect = cv2.minAreaRect(contour)
            width = min(rect[1])
            height = max(rect[1])
            aspect_ratio = float(width) / height if height != 0 else 0
            area = cv2.contourArea(contour)
            areas.append(area)

            if aspect_ratio >= min_aspect_ratio and (min_area is None or area >= min_area):
                filtered_contours.append(contour)
                filtered_holes.append(holes)

        self.contours_tissue = filtered_contours
        self.holes_tissue = filtered_holes

        return filtered_contours, filtered_holes

    def view_wsi(
        self,
        vis_level=5,
        color=(0, 255, 0),
        hole_color=(0, 0, 255),
        line_thickness=100,
        max_size=None,
        custom_downsample=1,
        view_slide_only=False,
        number_contours=False,
        seg_display=True,
    ):

        assert self.slide_image is not None, "Segment the WSI first."

        downsample = self.slide_image.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]

        top_left = (0, 0)
        region_size = self.slide_image.level_dim[vis_level]

        img = np.array(self.slide_image.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))

        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(
                        img,
                        self.scaleContourDim(self.contours_tissue, scale),
                        -1,
                        color,
                        line_thickness,
                        lineType=cv2.LINE_8,
                        offset=offset,
                    )
                else:  # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = 0, 0
                        # Adjust centroid based on offset
                        cX = int(cX * scale[0] + offset[0])
                        cY = int(cY * scale[1] + offset[1])
                        # draw the contour and put text next to center
                        cv2.drawContours(img, [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        print(f"Contour {idx}: Center at ({cX}, {cY})")

                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale), -1, hole_color, line_thickness, lineType=cv2.LINE_8)

        img = Image.fromarray(img)

        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype="int32") for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype="int32") for hole in holes] for holes in contours]

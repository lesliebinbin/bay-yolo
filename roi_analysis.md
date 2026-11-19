# ROI Analysis for Tank Truck Safety Monitoring

This document provides a high-level overview of which safety behaviors are monitored and whether they can be localized to a specific Region of Interest (ROI). Detailed coordinate configurations are stored in `src/config/item_rois.yaml`.

## Detection Items Mapping

| Item | Name | ROI Available | ROI Source | Temporal |
| :--- | :--- | :---: | :---: | :---: |
| **1** | **Personnel ID** | ✅ | A, B | No |
| **2** | **PPE: Helmet** | ✅ | A, B | No |
| **3** | **PPE: Gloves/Goggles** | ❌ | N/A | No |
| **4** | **Driver Location** | ✅ | A, B | No |
| **5** | **Prohibited Behavior** | ✅ | B | Yes |
| **6** | **Vehicle Verification** | ✅ | A, B | Yes |
| **7** | **Airtightness Check** | ✅ | B | Yes |
| **8** | **Emergency Valve** | ✅ | B | Yes |
| **9** | **Loading Inspection** | ✅ | A, B | Yes |
| **10** | **Post-Loading Evac** | ✅ | A, B | Yes |

---

## Legend
*   **ROI Source:** `A` = Video 01 (Overhead View), `B` = Video 02 (Side Profile View).
*   **Temporal:** `Yes` indicates the behavior requires a time sequence of frames to verify; `No` indicates it can be detected in a single frame.
*   **ROI Available:** Indicates if a stable region can be defined for detection given the current camera angles and resolution.

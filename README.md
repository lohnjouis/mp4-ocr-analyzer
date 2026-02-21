# MP4 OCR Analyzer
<img width="1163" height="789" alt="Screenshot 2026-02-21 033307" src="https://github.com/user-attachments/assets/7dbce889-b946-49a6-b326-3ff17f37178a" />


Turn a video of a digital gauge into a CSV dataset for Excel, plotting, and analysis.

This app is built for cases like:

- Speedometer footage
- Multimeter screens
- Panel readouts
- Any digital numeric display captured in video

You select a region of interest (ROI), sample frames, run OCR, and export values as a CSV.

## Purpose

The main goal is practical data extraction from videos of numeric displays. Instead of manually transcribing values frame by frame, this tool automates OCR and produces time/value output that can be graphed or analyzed in tools like Excel.

## Why DirectML Is a First-Class Option

This project supports `CPU`, `GPU (DirectML)`, and `GPU (CUDA)` profiles.

DirectML was added because CUDA dependencies are much heavier in many environments:

- DirectML profile: roughly ~1 GB of dependencies
- CUDA profile: roughly ~2.6 GB of dependencies

DirectML is often the better default for Windows users who want GPU acceleration without the CUDA package footprint.

## Features

- MP4 video loading and preview
- Interactive ROI drawing for OCR target area
- Trim range selection with dual-handle slider
- Sampling by frame count or seconds
- OCR confidence thresholding and numeric parsing controls
- Processor selection:
  - Auto (prefer DirectML, then CUDA, then CPU)
  - CPU
  - GPU (DirectML)
  - GPU (CUDA)
- X-axis options for CSV export:
  - `video_time_sec`
  - `time` (clock format `HH:MM:SS.xx`)
- Optional end-time mapping for sped-up/slowed footage:
  - `Use end time (for sped-up or slowed video)`
- CSV export with optional header row
- Run statistics (duration, pass rate, threshold misses)

## Platform and Environment

- Windows-focused workflow (`install.ps1` / `install.bat`)
- Python virtual environment created in project folder (`.venv`)
- Uses PySide6 + OpenCV + RapidOCR + ONNX Runtime backend packages

## Installation

From the project directory, run one of:

- PowerShell (recommended):
  - `.\install.ps1`
- Command Prompt:
  - `install.bat`

The installer will:

1. Create `.venv` if missing
2. Ensure `pip` exists in that environment
3. Prompt for dependency profile (`directml`, `cuda`, or `cpu`)
4. Install the matching requirements file

### Non-interactive install

Use one of:

- `.\install.ps1 -Profile directml`
- `.\install.ps1 -Profile cuda`
- `.\install.ps1 -Profile cpu`

## Running the App

Use the project venv interpreter:

`.\.venv\Scripts\python.exe main.py`

## Usage Workflow

1. Load a video using **Browse MP4**.
2. Draw the OCR ROI directly on the preview.
3. Set trim start/end with the trim slider.
4. Choose sampling mode:
   - Frames: every N frames
   - Seconds: every N seconds
5. Tune OCR options:
   - Confidence threshold
   - Include negative values
   - Include decimals
   - Only include numbers
6. Choose processing backend (Auto/CPU/DirectML/CUDA).
7. Configure X-axis:
   - `video_time_sec` for raw video timeline seconds
   - `time` for clock time output
8. If using `time`:
   - Set **Start time**
   - Optionally enable **Use end time (for sped-up or slowed video)**
   - If enabled, set **End time**
9. Click **Run OCR and Save CSV**.

## Time Mapping Behavior

When X-axis is `time`:

- If **Use end time** is unchecked:
  - Output time is `start + video_seconds`
- If **Use end time** is checked:
  - Output time is linearly mapped from start to end across the selected trim range
  - Useful for timelapse or slow-motion footage where playback speed does not match real-world elapsed time
  - If end time is earlier than start time, it is treated as crossing midnight (next day)

## CSV Output

The CSV contains:

- X-axis column (`video_time_sec` or `time`)
- `value` column (parsed numeric OCR result)

Notes:

- Rows are created per sampled frame
- If OCR fails or confidence is below threshold, `value` is blank for that row
- Header row is optional (toggle in UI)

## Current Limitations

- OCR target is a single manual ROI; no automatic tracking if the display moves
- Numeric parsing is optimized for digital-style numbers, not arbitrary text
- Video input workflow is MP4-oriented
- End-time mapping is linear across trim range (no nonlinear speed correction)
- No automatic calibration to embedded video metadata/timecode
- OCR accuracy depends heavily on source quality (blur, glare, compression, motion, perspective)
- GPU profile availability depends on installed runtime/provider support

## Recommendations

- Capture footage with:
  - Stable camera
  - Sharp focus
  - Minimal glare/reflection
  - High enough shutter speed to reduce motion blur
- Keep ROI tight around only the digits
- Use trim to exclude irrelevant portions
- Start with a slightly lower confidence threshold, then increase once stable
- For timelapse/slo-mo:
  - Use `time` axis
  - Enable **Use end time**
  - Enter known real-world start and end times
- Validate a subset of exported points against the video before relying on full-run analysis
- Prefer DirectML on Windows when CUDA install size/complexity is not justified

## Troubleshooting

### Windows opens Microsoft Store instead of running Python

- Disable App execution aliases for `python.exe`/`python3.exe` in Windows settings:
  - `Settings -> Apps -> Advanced app settings -> App execution aliases`
- Re-run `install.bat` or `install.ps1`

### Installer cannot find usable Python

- Install Python from python.org
- Ensure a real `python.exe` is available (not only Windows Store alias)
- Re-run the installer

### CUDA mode selected but unavailable

- Verify NVIDIA/CUDA environment and installed CUDA profile dependencies
- Use DirectML or CPU profile as fallback

### OCR misses values

- Improve ROI quality (contrast/sharpness)
- Adjust confidence threshold and numeric filters
- Increase sampling interval if processing too noisy or too dense

## Typical Use Cases

- Vehicle speed trace extraction from dash footage
- Lab instrument readout logging from recorded tests
- Power/voltage/current trend extraction from multimeter videos
- Any digital panel where values are visible but not natively logged

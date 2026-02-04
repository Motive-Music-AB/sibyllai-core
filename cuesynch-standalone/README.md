# CueSynch Standalone

A standalone tool for converting CSV marker lists to WAV files with embedded markers for import into Logic Pro, Adobe Audition, and other DAWs.

## Features

- Upload any CSV file with timecode data
- Auto-detect timecode column and frame rate
- Select which columns to include in marker names
- Generate WAV file with embedded markers
- Import markers directly into your DAW

## Files

- `src/components/CueSynch.tsx` - Main React component
- `src/lib/csv-parser.ts` - CSV parsing utilities
- `src/lib/wav-generator.ts` - WAV file generation with markers

## Usage

This component can be integrated into any React application. It requires:
- React 18+
- A Button component from your UI library (or replace with standard button)

## Supported Timecode Formats

- Seconds: `10.5`
- MM:SS: `01:30`
- HH:MM:SS: `01:30:45`
- HH:MM:SS:FF: `01:30:45:15` (with frames)

## Frame Rate Detection

Automatically detects frame rates: 24, 25, 30, 50, 60 fps based on timecode analysis.

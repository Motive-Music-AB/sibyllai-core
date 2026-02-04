/**
 * CSV Parsing Utilities for CueSynch
 * Ported from csv-to-logic web app
 */

export interface CSVRow {
  [key: string]: string;
}

export interface CSVData {
  headers: string[];
  rows: CSVRow[];
}

export interface Marker {
  time: number;
  name: string;
}

/**
 * Parse a single CSV line, handling quoted values
 */
export function parseCSVLine(line: string): string[] {
  const values: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      values.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  values.push(current.trim());
  return values;
}

/**
 * Parse CSV content into headers and rows
 */
export function parseCSVWithHeaders(csvContent: string): CSVData {
  const lines = csvContent.trim().split('\n');
  if (lines.length === 0) {
    throw new Error('CSV file is empty');
  }

  // Parse header line
  const headers = parseCSVLine(lines[0]);

  // Parse all data rows
  const rows: CSVRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const values = parseCSVLine(line);
    if (values.length > 0) {
      const row: CSVRow = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || '';
      });
      rows.push(row);
    }
  }

  return { headers, rows };
}

/**
 * Parse time string to seconds
 * Supports: seconds (10.5), MM:SS, HH:MM:SS, HH:MM:SS:FF
 */
export function parseTime(timeStr: string, frameRate: number = 30): number | null {
  timeStr = timeStr.trim();

  // Check if it contains colons (timecode format)
  if (!timeStr.includes(':')) {
    // Try to parse as seconds (e.g., "10.5" or "10")
    const seconds = parseFloat(timeStr);
    if (!isNaN(seconds)) {
      return seconds;
    }
    return null;
  }

  // Parse as MM:SS, HH:MM:SS, or HH:MM:SS:FF
  const timeParts = timeStr.split(':');
  if (timeParts.length === 2) {
    // MM:SS
    const minutes = parseInt(timeParts[0]);
    const secs = parseFloat(timeParts[1]);
    if (!isNaN(minutes) && !isNaN(secs)) {
      return minutes * 60 + secs;
    }
  } else if (timeParts.length === 3) {
    // HH:MM:SS
    const hours = parseInt(timeParts[0]);
    const minutes = parseInt(timeParts[1]);
    const secs = parseFloat(timeParts[2]);
    if (!isNaN(hours) && !isNaN(minutes) && !isNaN(secs)) {
      return hours * 3600 + minutes * 60 + secs;
    }
  } else if (timeParts.length === 4) {
    // HH:MM:SS:FF (with frames)
    const hours = parseInt(timeParts[0]);
    const minutes = parseInt(timeParts[1]);
    const secs = parseInt(timeParts[2]);
    const frames = parseInt(timeParts[3]);
    if (!isNaN(hours) && !isNaN(minutes) && !isNaN(secs) && !isNaN(frames)) {
      // Convert frames to seconds based on frame rate
      const frameSeconds = frames / frameRate;
      return hours * 3600 + minutes * 60 + secs + frameSeconds;
    }
  }

  return null;
}

/**
 * Smart frame rate detection based on CSV content
 */
export function detectFrameRate(rows: CSVRow[], timeColumn: string): number {
  const frameNumbers: number[] = [];

  // Collect all frame numbers from timecodes with 4 parts (HH:MM:SS:FF)
  for (const row of rows) {
    const timeStr = row[timeColumn]?.trim();
    if (!timeStr) continue;

    const timeParts = timeStr.split(':');
    if (timeParts.length === 4) {
      const frames = parseInt(timeParts[3]);
      if (!isNaN(frames)) {
        frameNumbers.push(frames);
      }
    }
  }

  // No frame-based timecodes found, return default
  if (frameNumbers.length === 0) {
    return 30;
  }

  // Smart detection based on maximum frame number
  const maxFrame = Math.max(...frameNumbers);

  if (maxFrame < 24) {
    return 24; // Film standard
  } else if (maxFrame < 25) {
    return 25; // PAL standard
  } else if (maxFrame < 30) {
    return 30; // NTSC / Common standard
  } else if (maxFrame < 50) {
    return 50; // High frame rate
  } else {
    return 60; // Very high frame rate
  }
}

/**
 * Title case a string (capitalize first letter of each word)
 */
function toTitleCase(str: string): string {
  return str.replace(/\b\w/g, char => char.toUpperCase());
}

/**
 * Add label prefix for specific fields with brackets
 */
function addFieldLabel(field: string, value: string): string {
  const fieldLower = field.toLowerCase();
  const capitalizedValue = toTitleCase(value);
  if (fieldLower === 'bpm') return `[BPM ${value}]`; // Keep BPM value as-is (it's a number)
  if (fieldLower === 'instruments') return `[Inst: ${capitalizedValue}]`;
  if (fieldLower === 'genres') return `[Genre: ${capitalizedValue}]`;
  if (fieldLower === 'style') return `[Style: ${capitalizedValue}]`;
  return capitalizedValue;
}

/**
 * Parse CSV with selected fields for marker names
 */
export function parseCSVWithSelectedFields(
  rows: CSVRow[],
  frameRate: number,
  timeColumn: string,
  selectedFields: string[]
): Marker[] {
  const markers: Marker[] = [];

  for (const row of rows) {
    const timeStr = row[timeColumn];
    if (!timeStr) continue;

    const time = parseTime(timeStr, frameRate);
    if (time === null) continue;

    // Build marker name from selected fields with labels
    const nameParts = selectedFields
      .map(field => {
        const value = row[field];
        if (!value || !value.trim()) return null;
        return addFieldLabel(field, value.trim());
      })
      .filter((value): value is string => value !== null);

    const name = nameParts.length > 0 ? nameParts.join(' - ') : 'Marker';

    markers.push({ time, name });
  }

  // Sort markers by time
  markers.sort((a, b) => a.time - b.time);

  return markers;
}

/**
 * Auto-detect time column from headers
 */
export function detectTimeColumn(headers: string[]): string {
  const timeKeywords = ['time', 'timecode', 'tc', 'smpte', 'timestamp'];

  for (const header of headers) {
    const headerLower = header.toLowerCase();
    if (timeKeywords.some(keyword => headerLower.includes(keyword))) {
      return header;
    }
  }

  // Default to first column if no match
  return headers[0];
}

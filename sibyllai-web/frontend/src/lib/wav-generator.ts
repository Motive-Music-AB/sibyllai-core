/**
 * WAV File Generator with Embedded Markers
 * Browser-compatible version using ArrayBuffer
 * Generates BWF (Broadcast Wave Format) compliant files for Logic Pro
 */

import type { Marker } from './csv-parser';

/**
 * Generate a WAV file Blob with embedded markers (cue points)
 * @param markers Array of markers with time and name
 * @param sampleRate Sample rate in Hz (44100, 48000, or 96000)
 */
export function generateWavWithMarkers(markers: Marker[], sampleRate: number = 48000): Blob {
  // Audio parameters
  const numChannels = 2;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;

  // Calculate duration (last marker time + 1 second)
  const duration = markers.length > 0 ? Math.ceil(markers[markers.length - 1].time) + 1 : 10;
  const numSamples = duration * sampleRate;

  // Create chunks
  const fmtChunk = createFmtChunk(sampleRate, numChannels, bitsPerSample);
  const bextChunk = createBextChunk();
  const cueChunk = createCueChunk(markers, sampleRate);
  const listChunk = createListChunk(markers);
  const dataChunk = createDataChunk(numSamples, numChannels, bytesPerSample);

  // Calculate total file size
  const fileSize = 4 + // 'WAVE'
    8 + fmtChunk.byteLength + // 'fmt ' + size + data
    8 + bextChunk.byteLength + // 'bext' + size + data
    8 + dataChunk.byteLength + // 'data' + size + data
    8 + cueChunk.byteLength + // 'cue ' + size + data
    8 + listChunk.byteLength; // 'LIST' + size + data

  // Create RIFF header
  const riffHeader = new ArrayBuffer(12);
  const riffView = new DataView(riffHeader);
  writeString(riffView, 0, 'RIFF');
  riffView.setUint32(4, fileSize, true);
  writeString(riffView, 8, 'WAVE');

  // Combine all chunks
  const chunks = [
    riffHeader,
    createChunkHeader('fmt ', fmtChunk.byteLength), fmtChunk,
    createChunkHeader('bext', bextChunk.byteLength), bextChunk,
    createChunkHeader('data', dataChunk.byteLength), dataChunk,
    createChunkHeader('cue ', cueChunk.byteLength), cueChunk,
    createChunkHeader('LIST', listChunk.byteLength), listChunk
  ];

  return new Blob(chunks, { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function createChunkHeader(id: string, size: number): ArrayBuffer {
  const header = new ArrayBuffer(8);
  const view = new DataView(header);
  writeString(view, 0, id);
  view.setUint32(4, size, true);
  return header;
}

function createFmtChunk(sampleRate: number, numChannels: number, bitsPerSample: number): ArrayBuffer {
  const chunk = new ArrayBuffer(16);
  const view = new DataView(chunk);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const byteRate = sampleRate * blockAlign;

  view.setUint16(0, 1, true); // Audio format (1 = PCM)
  view.setUint16(2, numChannels, true); // Number of channels
  view.setUint32(4, sampleRate, true); // Sample rate
  view.setUint32(8, byteRate, true); // Byte rate
  view.setUint16(12, blockAlign, true); // Block align
  view.setUint16(14, bitsPerSample, true); // Bits per sample

  return chunk;
}

function createBextChunk(): ArrayBuffer {
  // Broadcast Wave Format Extension chunk (602 bytes minimum)
  const chunk = new ArrayBuffer(602);
  const view = new DataView(chunk);
  const uint8View = new Uint8Array(chunk);

  // Description (256 bytes)
  const description = 'Motive CueSynch - Generated marker file';
  for (let i = 0; i < description.length && i < 256; i++) {
    uint8View[i] = description.charCodeAt(i);
  }

  // Originator (32 bytes) at offset 256
  const originator = 'Motive CueSynch';
  for (let i = 0; i < originator.length && i < 32; i++) {
    uint8View[256 + i] = originator.charCodeAt(i);
  }

  // OriginatorReference (32 bytes) at offset 288
  const originatorRef = `MOTIVE${Date.now()}`;
  for (let i = 0; i < originatorRef.length && i < 32; i++) {
    uint8View[288 + i] = originatorRef.charCodeAt(i);
  }

  // OriginationDate (10 bytes) at offset 320 - format: yyyy:mm:dd
  const date = new Date();
  const originationDate = `${date.getFullYear()}:${String(date.getMonth() + 1).padStart(2, '0')}:${String(date.getDate()).padStart(2, '0')}`;
  for (let i = 0; i < originationDate.length; i++) {
    uint8View[320 + i] = originationDate.charCodeAt(i);
  }

  // OriginationTime (8 bytes) at offset 330 - format: hh:mm:ss
  const originationTime = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
  for (let i = 0; i < originationTime.length; i++) {
    uint8View[330 + i] = originationTime.charCodeAt(i);
  }

  // TimeReference (8 bytes: low 4 bytes + high 4 bytes) at offset 338
  // TimeReference = 0 means the file starts at timeline position 00:00:00:00
  view.setUint32(338, 0, true);
  view.setUint32(342, 0, true);

  // Version (2 bytes) at offset 346 - BWF version
  view.setUint16(346, 2, true);

  return chunk;
}

function createCueChunk(markers: Marker[], sampleRate: number): ArrayBuffer {
  const numCuePoints = markers.length;
  const chunk = new ArrayBuffer(4 + numCuePoints * 24);
  const view = new DataView(chunk);

  // Write number of cue points
  view.setUint32(0, numCuePoints, true);

  // Write each cue point
  let offset = 4;
  markers.forEach((marker, index) => {
    const samplePosition = Math.floor(marker.time * sampleRate);

    view.setUint32(offset, index + 1, true); // Cue point ID
    view.setUint32(offset + 4, samplePosition, true); // Position
    writeString(view, offset + 8, 'data'); // Data chunk ID
    view.setUint32(offset + 12, 0, true); // Chunk start
    view.setUint32(offset + 16, 0, true); // Block start
    view.setUint32(offset + 20, samplePosition, true); // Sample offset

    offset += 24;
  });

  return chunk;
}

function createListChunk(markers: Marker[]): ArrayBuffer {
  // Calculate total size needed for all label subchunks
  let totalSize = 4; // 'adtl' type identifier

  const subChunkSizes: number[] = [];
  markers.forEach((marker) => {
    const labelBytes = new TextEncoder().encode(marker.name);
    const labelLength = labelBytes.length + 1; // +1 for null terminator
    const paddedLength = labelLength + (labelLength % 2); // Pad to even length
    const subChunkSize = 8 + 4 + paddedLength; // header + cue ID + label
    subChunkSizes.push(subChunkSize);
    totalSize += subChunkSize;
  });

  const chunk = new ArrayBuffer(totalSize);
  const view = new DataView(chunk);
  const uint8View = new Uint8Array(chunk);

  // Write 'adtl' list type
  writeString(view, 0, 'adtl');

  // Write each label subchunk
  let offset = 4;
  markers.forEach((marker, index) => {
    const labelBytes = new TextEncoder().encode(marker.name);
    const labelLength = labelBytes.length + 1;
    const paddedLength = labelLength + (labelLength % 2);

    // Write 'labl' subchunk ID
    writeString(view, offset, 'labl');
    // Write subchunk size
    view.setUint32(offset + 4, 4 + paddedLength, true);
    // Write cue point ID
    view.setUint32(offset + 8, index + 1, true);
    // Write label text
    uint8View.set(labelBytes, offset + 12);
    // Null terminator is already 0 (ArrayBuffer is initialized to zeros)

    offset += subChunkSizes[index];
  });

  return chunk;
}

function createDataChunk(numSamples: number, numChannels: number, bytesPerSample: number): ArrayBuffer {
  // Create silent audio data
  const dataSize = numSamples * numChannels * bytesPerSample;
  return new ArrayBuffer(dataSize); // All zeros = silence
}

/**
 * Download a blob as a file
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

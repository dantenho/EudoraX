
import { GoogleGenAI, Modality } from "@google/genai";

/**
 * Creates a new instance of GoogleGenAI using the current API_KEY.
 * This ensures the most up-to-date key is used for each request.
 */
const getAI = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) throw new Error("API_KEY_REQUIRED");
  return new GoogleGenAI({ apiKey });
};

/**
 * Decoding helper: base64 string to Uint8Array.
 */
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

/**
 * Decoding helper: raw PCM data to AudioBuffer.
 * The Gemini TTS API returns raw PCM 16-bit audio.
 */
async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

/**
 * Enhanced Image Synthesis with Thinking & Search
 */
export const synthesizeImage = async (params: {
  prompt: string;
  aspectRatio: string;
  useThinking?: boolean;
  useSearch?: boolean;
  referenceImage?: string; // Base64
}) => {
  const ai = getAI();
  // gemini-3-pro-image-preview is used when search grounding is required.
  const model = params.useSearch ? 'gemini-3-pro-image-preview' : 'gemini-2.5-flash-image';

  const parts: any[] = [{ text: params.prompt }];
  
  if (params.referenceImage) {
    parts.unshift({
      inlineData: {
        mimeType: 'image/png',
        data: params.referenceImage.split(',')[1]
      }
    });
  }

  const response = await ai.models.generateContent({
    model,
    contents: { parts },
    config: {
      imageConfig: { 
        aspectRatio: params.aspectRatio as any,
        imageSize: model === 'gemini-3-pro-image-preview' ? '1K' : undefined
      },
      thinkingConfig: params.useThinking ? { thinkingBudget: 1000 } : undefined,
      tools: params.useSearch ? [{ googleSearch: {} }] : undefined
    }
  });

  let imageUrl = null;
  let searchLinks: any[] = [];

  for (const part of response.candidates?.[0]?.content?.parts || []) {
    if (part.inlineData) {
      imageUrl = `data:image/png;base64,${part.inlineData.data}`;
    }
  }

  if (response.candidates?.[0]?.groundingMetadata?.groundingChunks) {
    searchLinks = response.candidates[0].groundingMetadata.groundingChunks;
  }

  return { imageUrl, searchLinks, text: response.text };
};

/**
 * Generate Video using Veo.
 * Supports text prompt and returns a signed download link.
 */
export const generateVideo = async (prompt: string): Promise<string> => {
  const ai = getAI();
  let operation = await ai.models.generateVideos({
    model: 'veo-3.1-fast-generate-preview',
    prompt: prompt,
    config: {
      numberOfVideos: 1,
      resolution: '1080p',
      aspectRatio: '16:9'
    }
  });
  
  // Wait for the long-running operation to complete.
  while (!operation.done) {
    await new Promise(resolve => setTimeout(resolve, 5000));
    operation = await ai.operations.getVideosOperation({ operation: operation });
  }

  const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
  if (!downloadLink) throw new Error("Video generation failed: No URI returned.");
  
  // The response body contains the MP4 bytes. You must append an API key when fetching from the download link.
  return `${downloadLink}&key=${process.env.API_KEY}`;
};

/**
 * Generate Speech using Gemini 2.5 TTS model.
 * Returns an AudioBuffer for immediate playback.
 */
export const generateSpeech = async (text: string, voiceName: string): Promise<AudioBuffer | null> => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: [{ parts: [{ text }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: voiceName as any },
        },
      },
    },
  });

  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64Audio) return null;

  // Audio decoding context (24kHz as per model output).
  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
  
  return await decodeAudioData(
    decode(base64Audio),
    audioContext,
    24000,
    1
  );
};

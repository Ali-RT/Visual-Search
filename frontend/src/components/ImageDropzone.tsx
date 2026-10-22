import { useCallback, useState } from 'react';
import { Upload, X } from 'lucide-react';

interface Props {
  onFileSelect: (file: File) => void;
  preview?: string | null;
  onClear?: () => void;
}

export function ImageDropzone({ onFileSelect, preview, onClear }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('image/')) onFileSelect(file);
    },
    [onFileSelect],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect],
  );

  if (preview) {
    return (
      <div className="relative rounded-xl overflow-hidden border border-gray-700 bg-gray-900">
        <img src={preview} alt="Query" className="w-full max-h-[560px] object-contain" />
        {onClear && (
          <button
            onClick={onClear}
            className="absolute top-2 right-2 p-1.5 bg-gray-900/80 rounded-full hover:bg-red-600 transition-colors"
          >
            <X size={16} />
          </button>
        )}
      </div>
    );
  }

  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`flex flex-col items-center justify-center gap-3 p-10 rounded-xl border-2 border-dashed cursor-pointer transition-all ${
        dragOver
          ? 'border-blue-400 bg-blue-950/30'
          : 'border-gray-700 bg-gray-900/50 hover:border-gray-500'
      }`}
    >
      <Upload size={36} className="text-gray-500" />
      <p className="text-sm text-gray-400">
        Drag & drop an image or <span className="text-blue-400 underline">browse</span>
      </p>
      <input type="file" accept="image/*" onChange={handleChange} className="hidden" />
    </label>
  );
}

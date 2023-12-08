export type ModelSourceItem = {
  name: string;
  size: number;
  lastUpdated: string;
  desc?: { [lang: string]: string | undefined; };
  SHA256?: string;
  url?: string;
  downloadUrl?: string;
  isComplete?: boolean;
  isLocal?: boolean;
  localSize?: number;
  lastUpdatedMs?: number;
  tags?: string[];
  hide?: boolean;
};
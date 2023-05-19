export type Record = {
  question: string;
  answer: string;
}

export type ConversationPair = {
  role: string;
  content: string;
}

export function getConversationPairs(records: Record[], isCompletion: boolean): string | ConversationPair[] {
  let pairs;
  if (isCompletion) {
    pairs = '';
    for (const record of records) {
      pairs += 'Human: ' + record.question + '\nAI: ' + record.answer + '\n';
    }
  } else {
    pairs = [];
    for (const record of records) {
      pairs.push({role: 'user', content: record.question});
      pairs.push({role: 'assistant', content: record.answer});
    }
  }

  return pairs;
}

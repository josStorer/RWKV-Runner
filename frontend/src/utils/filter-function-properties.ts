export type FilterFunctionProperties<T> = {
  // eslint-disable-next-line
  [K in keyof T as T[K] extends Function ? never : K]: T[K]
}

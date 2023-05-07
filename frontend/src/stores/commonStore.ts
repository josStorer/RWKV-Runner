import {makeAutoObservable} from 'mobx';

export enum ModelStatus {
  Offline,
  Starting,
  Loading,
  Working,
}

class CommonStore {
  constructor() {
    makeAutoObservable(this);
  }

  modelStatus: ModelStatus = ModelStatus.Offline;
  updateModelStatus = (status: ModelStatus) => {
    this.modelStatus = status;
  };
}

export default new CommonStore();
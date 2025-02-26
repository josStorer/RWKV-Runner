import { makeAutoObservable } from 'mobx'
import { v4 as uuid } from 'uuid'
import { IsCmdRunning } from '../../wailsjs/go/backend_golang/App'
import { TaskResult } from '../utils/rwkv-task'

type TaskFunction = (...args: any[]) => Promise<TaskResult>

export interface Task<F extends TaskFunction = TaskFunction> {
  name?: string
  func: F
  args: Parameters<F> extends [...infer P, (output: string) => void]
    ? P
    : Parameters<F>
}

type TaskChainStatus = 'pending' | 'running' | 'success' | 'error' | 'cancelled'

export interface TaskChain {
  id: string
  name: string
  createdAt: number
  updatedAt: number
  currentTaskIndex: number
  status: TaskChainStatus
  lines: string[]
  tasks: Task[]
}

class CmdTaskChainStore {
  activeTaskChainId: string | null = null
  taskChainIds: string[] = []
  taskChainMap: Record<string, TaskChain> = {}

  constructor() {
    makeAutoObservable(this)
  }

  newTaskChain<F extends TaskFunction>(name: string, tasks: Task<F>[]) {
    const now = Date.now()
    const id = uuid()
    const taskChain: TaskChain = {
      id,
      name,
      createdAt: now,
      updatedAt: now,
      currentTaskIndex: 0,
      status: 'pending',
      lines: [],
      tasks,
    }
    this.taskChainMap[id] = taskChain
    this.taskChainIds.push(id)
    return id
  }

  async startTaskChain(
    id: string,
    finishPredicate?: (message: string) => boolean
  ) {
    const taskChain = this.taskChainMap[id]
    if (!taskChain) {
      throw new Error('Task chain not found')
    }
    this.activeTaskChainId = id
    taskChain.status = 'running'
    for (const task of taskChain.tasks) {
      try {
        const result = await task.func(...task.args, (output: string) => {
          taskChain.lines.push(output)
          taskChain.updatedAt = Date.now()
          if (finishPredicate?.(output)) {
            taskChain.status = 'success'
            return true
          }
        })
      } catch (e) {
        taskChain.status = 'error'
        return false
      }
      taskChain.currentTaskIndex++
    }
    taskChain.status = 'success'
    return true
  }
}

export default new CmdTaskChainStore()

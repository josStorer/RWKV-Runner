import { ReactNode } from 'react'
import delay from 'delay'
import { makeAutoObservable } from 'mobx'
import { v4 as uuid } from 'uuid'
import { TaskResult } from '../utils/rwkv-task'

type TaskFunction = (...args: any[]) => Promise<TaskResult>

/** if the task continue to execute, return true, otherwise return false */
export type OutputHandler = (output: string) => boolean

export interface Task<F extends TaskFunction = TaskFunction> {
  /** used to show in the UI */
  name?: string
  /** should be unique, it will be used to jump to the task */
  id?: string
  func: F
  args: Parameters<F> extends [...infer P, OutputHandler] ? P : Parameters<F>
  /**
   * predicate to jump to other task, useful when error occurs
   * @param message - the output of the task
   * @param jumpTo - a function to jump to another task, call it by yourself. priority is higher than jumpId.
   * when use jumpTo, you should always return false.
   */
  jumpPredicate?: (message: string, jumpTo: (jumpId: string) => void) => boolean
  /** jump to the task if jumpPredicate is true.
   * if jumpPredicate is not provided, still jump when task is finished.
   * cannot jump to self.
   */
  jumpId?: string
  /** if true, new output will override the previous output in the same line instead of adding a new line */
  refreshLine?: boolean
  /** define how to render the message */
  lineRenderer?: (message: string) => ReactNode
}

export type TaskChainStatus =
  | 'pending'
  | 'running'
  | 'success'
  | 'error'
  | 'cancelled'

export interface TaskChain {
  id: string
  name: string
  createdAt: number
  updatedAt: number
  currentExecuteTaskIndex: number
  status: TaskChainStatus
  executeTaskInfo: {
    executeId: string
    lines: (ReactNode | string)[]
  }[]
  tasksExecuteIdOrder: string[]
  /** tasks will be executed in order */
  tasks: { [executeId: string]: Task }
  /** helper tasks could be jumped to */
  helperTasks: { [executeId: string]: Task }
}

const getCurrentTaskExecuteId = (taskChain: TaskChain) => {
  return taskChain.executeTaskInfo[taskChain.currentExecuteTaskIndex].executeId
}

const getTaskByExecuteId = (taskChain: TaskChain, executeId: string) => {
  return taskChain.tasks[executeId] || taskChain.helperTasks[executeId]
}

class CmdTaskChainStore {
  consoleOpen: boolean = false
  activeTaskChainId: string | null = null
  taskChainIds: string[] = []
  taskChainMap: Record<string, TaskChain> = {}
  taskChainCurrentTaskMap: Record<string, TaskResult> = {}

  constructor() {
    makeAutoObservable(this)
  }

  setConsoleOpen(open: boolean) {
    this.consoleOpen = open
  }

  newTaskChain<T extends TaskFunction[], H extends TaskFunction[]>(
    name: string,
    tasks: { [K in keyof T]: Task<T[K]> },
    helperTasks?: { [K in keyof H]: Task<H[K]> }
  ) {
    const now = Date.now()
    const id = uuid()
    const tasksExecuteIdOrder = tasks.map(() => uuid())
    const taskChain: TaskChain = {
      id,
      name,
      createdAt: now,
      updatedAt: now,
      currentExecuteTaskIndex: 0,
      status: 'pending',
      executeTaskInfo: [],
      tasksExecuteIdOrder,
      tasks: Object.fromEntries(
        tasksExecuteIdOrder.map((id, i) => [id, tasks[i]])
      ),
      helperTasks: Object.fromEntries(
        (helperTasks || []).map((task) => [uuid(), task])
      ),
    }
    this.taskChainMap[id] = taskChain
    this.taskChainIds.push(id)
    return id
  }

  showAvailableTaskChain(excludedTaskChainIds: string[] = []) {
    const availableTaskChainId = this.taskChainIds.find((id) => {
      const taskChain = this.taskChainMap[id]
      return (
        taskChain.status === 'running' && !excludedTaskChainIds.includes(id)
      )
    })
    if (availableTaskChainId) {
      this.activeTaskChainId = availableTaskChainId
    }
  }

  /** status will be updated, and _afterFinishTaskChain will be called in the start loop */
  stopTaskChain(id: string) {
    this.taskChainCurrentTaskMap[id]?.stop()
  }

  _afterFinishTaskChain(id: string) {
    delete this.taskChainCurrentTaskMap[id]
  }

  deleteTaskChain(id: string) {
    this.stopTaskChain(id)
    delete this.taskChainCurrentTaskMap[id]
    delete this.taskChainMap[id]
    this.taskChainIds = this.taskChainIds.filter((_id) => _id !== id)
    if (this.activeTaskChainId === id) {
      this.activeTaskChainId = null
      this.showAvailableTaskChain()
    }
  }

  async startTaskChain(
    id: string,
    finishPredicate?: (message: string) => boolean
  ) {
    const taskChain = this.taskChainMap[id]
    if (!taskChain) {
      throw new Error('Task chain not found')
    }
    this.setConsoleOpen(true)
    this.activeTaskChainId = id

    taskChain.status = 'running'
    taskChain.currentExecuteTaskIndex = 0
    taskChain.executeTaskInfo.push({
      executeId: taskChain.tasksExecuteIdOrder[0],
      lines: [],
    })

    const jumpTasks = {
      ...taskChain.tasks,
      ...taskChain.helperTasks,
    }
    while (true) {
      if (taskChain.status !== 'running') {
        this._afterFinishTaskChain(id)
        return
      }

      const currentExecuteTaskIndex = taskChain.currentExecuteTaskIndex
      const currentExecuteTaskId = getCurrentTaskExecuteId(taskChain)
      const task = getTaskByExecuteId(taskChain, currentExecuteTaskId)
      let func: TaskResult | undefined
      const outputHandler: OutputHandler = (output) => {
        const postOutput = task.lineRenderer
          ? task.lineRenderer(output)
          : output
        if (task.refreshLine) {
          taskChain.executeTaskInfo[currentExecuteTaskIndex].lines = [
            postOutput,
          ]
        } else {
          taskChain.executeTaskInfo[currentExecuteTaskIndex].lines.push(
            postOutput
          )
        }

        taskChain.updatedAt = Date.now()

        if (finishPredicate?.(output)) {
          taskChain.status = 'success'
          func?.stop()
          return true
        }

        if (task.jumpPredicate?.(output, jumpTo)) {
          const jumpTask = Object.entries(jumpTasks).find(
            ([_, t]) => t.id && t.id === task.jumpId
          )
          if (jumpTask) {
            taskChain.executeTaskInfo.push({
              executeId: jumpTask[0],
              lines: [],
            })
            taskChain.currentExecuteTaskIndex++
            func?.stop()
            return true
          }
        }
        return false
      }

      const jumpTo = (jumpId: string) => {
        const jumpTask = Object.entries(jumpTasks).find(
          ([_, t]) => t.id && t.id === jumpId
        )
        if (jumpTask) {
          taskChain.executeTaskInfo.push({
            executeId: jumpTask[0],
            lines: [],
          })
          taskChain.currentExecuteTaskIndex++
          runContinue = true
          func?.stop()
        }
      }

      let runContinue = false
      func = await task.func(...task.args, outputHandler).catch((e) => {
        console.log('TaskChain, js task error handler:', e)
        const tempRunContinue = outputHandler(e.toString())
        // runContinue could be true when jumpTo is called, which is in jumpPredicate in outputHandler
        if (!runContinue) {
          runContinue = tempRunContinue
        }
        if (!runContinue) {
          taskChain.status = 'error'
          throw e
        } else {
          // just for type inference
          return void 0 as unknown as TaskResult
        }
      })
      this.taskChainCurrentTaskMap[id] = func
      if (runContinue) {
        continue
      }

      const isStopped = !(await func.promise.catch((e) => {
        console.log('TaskChain, cmd task error handler:', e)
        const tempRunContinue = outputHandler(e.toString())
        // runContinue could be true when jumpTo is called, which is in jumpPredicate in outputHandler
        if (!runContinue) {
          runContinue = tempRunContinue
        }
        if (!runContinue) {
          taskChain.status = 'error'
          throw e
        } else {
          // just for type inference
          return void 0 as unknown as boolean
        }
      }))
      if (runContinue) {
        continue
      }

      if (
        isStopped &&
        taskChain.currentExecuteTaskIndex === currentExecuteTaskIndex
      ) {
        if (taskChain.status === 'running') {
          taskChain.status = 'cancelled'
          this._afterFinishTaskChain(id)
        }
        return
      }

      if (task.jumpPredicate && task.jumpId) {
        // if task is finished and jumpPredicate is satisfied at the same time, we need to wait for jump
        await delay(500)
        if (taskChain.currentExecuteTaskIndex !== currentExecuteTaskIndex) {
          continue
        }
      }
      if (!task.jumpPredicate && task.jumpId) {
        const jumpTask = Object.entries(jumpTasks).find(
          ([_, t]) => t.id && t.id === task.jumpId
        )
        if (jumpTask) {
          taskChain.executeTaskInfo.push({
            executeId: jumpTask[0],
            lines: [],
          })
          taskChain.currentExecuteTaskIndex++
          continue
        }
      }

      // go next task normally
      const currentNormalTaskIndex =
        taskChain.tasksExecuteIdOrder.indexOf(currentExecuteTaskId)
      if (currentNormalTaskIndex === -1) {
        this._afterFinishTaskChain(id)
        throw new Error(`task cannot continue at ${task.func.name}`)
      }

      if (
        taskChain.currentExecuteTaskIndex >=
        taskChain.tasksExecuteIdOrder.length - 1
      ) {
        taskChain.status = 'success'
        this._afterFinishTaskChain(id)
        return
      } else {
        const nextTaskId =
          taskChain.tasksExecuteIdOrder[currentNormalTaskIndex + 1]
        taskChain.executeTaskInfo.push({
          executeId: nextTaskId,
          lines: [],
        })
        taskChain.currentExecuteTaskIndex++
      }
    }
  }
}

export default new CmdTaskChainStore()

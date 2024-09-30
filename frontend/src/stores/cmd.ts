import { observable } from 'mobx'
import { EventsOn } from '../../wailsjs/runtime/runtime'

interface Task {
  name: string
  timestamp: number
  lines: string[]
}

class CmdStore {
  handle: (() => void) | null = null

  taskMap = observable.map<string, Task>({})

  constructor() {}

  registerEvent() {
    this.handle = EventsOn('cmd_event', this._onCmdEvent)
  }

  _onCmdEvent = (...event: any) => {
    console.log('🚀 _onCmdEvent', event)
    // taskName_timestamp
    const threadID = event[0]
    const taskName = threadID.split('_')[0]
    const timestamp = threadID.split('_')[1]
    const strLine = event[1]
    console.log('🚀 _onCmdEvent', strLine)
    const task = this.taskMap.get(taskName)
    if (task) {
      task.lines.push(strLine)
    } else {
      this.taskMap.set(taskName, {
        name: taskName,
        timestamp,
        lines: [strLine],
      })
    }
  }

  unregisterEvent() {
    if (this.handle) {
      this.handle()
      this.handle = null
    }
  }
}

export default new CmdStore()

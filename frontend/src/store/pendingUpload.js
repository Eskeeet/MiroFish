/**
 * 临时存储待上传的文件和需求
 * 用于首页点击启动引擎后立即跳转，在Process页面再进行API调用
 */
import { reactive } from 'vue'

const state = reactive({
  files: [],
  localPaths: '',
  simulationRequirement: '',
  isPending: false
})

export function setPendingUpload(files, requirement, localPaths = '') {
  state.files = files
  state.localPaths = localPaths
  state.simulationRequirement = requirement
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    localPaths: state.localPaths,
    simulationRequirement: state.simulationRequirement,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.localPaths = ''
  state.simulationRequirement = ''
  state.isPending = false
}

export default state

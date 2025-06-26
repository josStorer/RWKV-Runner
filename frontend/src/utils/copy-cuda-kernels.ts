import { CopyFolderFiles } from '../../wailsjs/go/backend_golang/App'

export async function copyCudaKernels(torchVersion: string) {
  const copyRoot = './backend-python/rwkv_pip'
  if (torchVersion === '2.7.1+cu128') {
    await CopyFolderFiles(
      copyRoot + '/kernels/torch-2.7.1+cu128',
      copyRoot,
      true
    )
  } else {
    await CopyFolderFiles(
      copyRoot + '/kernels/torch-1.13.1+cu117',
      copyRoot,
      true
    )
  }
}

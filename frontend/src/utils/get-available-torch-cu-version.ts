import { compare } from 'compare-versions'

export const torchVersions = ['1.13.1', '2.7.1']

export function getAvailableTorchCuVersion(
  torchVersion: string,
  driverCudaVersion: string
) {
  let retTorchVersion = ''
  let retCuSourceVersion = ''
  const targetTorchVersion = torchVersion.split('+')[0]
  if (compare(targetTorchVersion, '2.7.1', '>=')) {
    retTorchVersion = '2.7.1'
    if (compare(driverCudaVersion, '12.8', '>=')) {
      retCuSourceVersion = '12.8'
    } else if (compare(driverCudaVersion, '12.6', '>=')) {
      retCuSourceVersion = '12.6'
    } else {
      retCuSourceVersion = '11.8'
    }
  } else {
    retTorchVersion = '1.13.1'
    if (compare(driverCudaVersion, '11.7', '>=')) {
      retCuSourceVersion = '11.7'
    } else {
      retCuSourceVersion = '11.6'
    }
  }
  return { torchVersion: retTorchVersion, cuSourceVersion: retCuSourceVersion }
}

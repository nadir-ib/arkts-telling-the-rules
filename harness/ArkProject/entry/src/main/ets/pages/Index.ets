@Entry
@Component
struct Index {
  @State message: string = 'Hello';
  build() {
    Column() { Text(this.message).fontSize(24) }
    .width('100%').height('100%')
  }
}

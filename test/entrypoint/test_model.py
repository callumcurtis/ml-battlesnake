from entrypoint import model


def test_snake_properties():
    name = "name"
    path = "path"
    snake = model.Snake(name, path)
    assert snake.name == name
    assert snake.path == path


def test_snake_str():
    name = "name"
    path = "path"
    snake = model.Snake(name, path)
    assert str(snake) == f"Snake(name={name}, path={path})"
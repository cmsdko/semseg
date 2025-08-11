# --- Build Stage ---
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Кладём весь проект
COPY . .

# Переходим в пример — тут и надо править go.mod
WORKDIR /app/example

# ВКЛ./ВЫКЛ. локального replace (на время локальной сборки из исходников)
# Если собираешь против релиза из GitHub — просто закомментируй следующую строку.
RUN printf '\nreplace github.com/cmsdko/semseg => ../\n' >> go.mod && go mod tidy

# Сборка бинаря
RUN CGO_ENABLED=0 go build -o /out/example .

# --- Final Stage ---
FROM alpine:latest
WORKDIR /root/
COPY --from=builder /out/example .
EXPOSE 8080
CMD ["./example"]

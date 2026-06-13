package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"strings"
	"sync"
)

type Todo struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Completed bool   `json:"completed"`
}

var (
	todos = make([]Todo, 0)
	mu    sync.Mutex
)

func main() {
	http.HandleFunc("/todos", todosHandler)
	http.HandleFunc("/todos/", todoByIDHandler)
	http.Handle("/", http.FileServer(http.Dir("./static")))
	log.Println("Server starting on :8080...")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}

func todosHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		getTodos(w, r)
	case "POST":
		addTodo(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func todoByIDHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/todos/")
	if id == "" {
		http.Error(w, "missing id", http.StatusBadRequest)
		return
	}
	switch r.Method {
	case "PUT":
		updateTodo(w, r, id)
	case "DELETE":
		deleteTodo(w, r, id)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func getTodos(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(todos)
}

func addTodo(w http.ResponseWriter, r *http.Request) {
	var todo Todo
	err := json.NewDecoder(r.Body).Decode(&todo)
	if err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	mu.Lock()
	todo.ID = fmt.Sprintf("%d", rand.Intn(100000))
	todos = append(todos, todo)
	mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(todo)
}

func updateTodo(w http.ResponseWriter, r *http.Request, id string) {
	var upd Todo
	err := json.NewDecoder(r.Body).Decode(&upd)
	if err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	mu.Lock()
	defer mu.Unlock()
	for i, t := range todos {
		if t.ID == id {
			if upd.Title != "" {
				todos[i].Title = upd.Title
			}
			todos[i].Completed = upd.Completed
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(todos[i])
			return
		}
	}
	http.Error(w, "not found", http.StatusNotFound)
}

func deleteTodo(w http.ResponseWriter, r *http.Request, id string) {
	mu.Lock()
	defer mu.Unlock()
	for i, t := range todos {
		if t.ID == id {
			todos = append(todos[:i], todos[i+1:]...)
			w.WriteHeader(http.StatusNoContent)
			return
		}
	}
	http.Error(w, "not found", http.StatusNotFound)
}

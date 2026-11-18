import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";

export const keys = {
  collections: ["collections"] as const,
  documents: ["documents"] as const,
  conversations: ["conversations"] as const,
  conversation: (id: string) => ["conversations", id] as const,
  page: (documentId: string, page: number) => ["page", documentId, page] as const,
  models: ["models"] as const,
  indexStatus: ["index-status"] as const,
};

export const useCollections = () =>
  useQuery({ queryKey: keys.collections, queryFn: api.listCollections });

export const useDocuments = () =>
  useQuery({ queryKey: keys.documents, queryFn: api.listDocuments });

export const useConversations = () =>
  useQuery({ queryKey: keys.conversations, queryFn: api.listConversations });

export const useConversation = (id: string | undefined) =>
  useQuery({
    queryKey: keys.conversation(id ?? ""),
    queryFn: () => api.getConversation(id!),
    enabled: Boolean(id),
  });

export const usePage = (documentId: string, page: number) =>
  useQuery({
    queryKey: keys.page(documentId, page),
    queryFn: () => api.getPage(documentId, page),
    staleTime: Infinity,
  });

// Listing models contacts the providers (Ollama, cloud API), so cache it for a while.
export const useModels = () =>
  useQuery({ queryKey: keys.models, queryFn: api.listModels, staleTime: 60_000 });

export const useIndexStatus = () =>
  useQuery({ queryKey: keys.indexStatus, queryFn: api.indexStatus });

/** Collection membership changes affect both the document list and collection counts. */
export function useLibraryInvalidation() {
  const queryClient = useQueryClient();
  return () =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: keys.documents }),
      queryClient.invalidateQueries({ queryKey: keys.collections }),
      queryClient.invalidateQueries({ queryKey: keys.indexStatus }),
    ]);
}

export function useMembershipMutation() {
  const invalidate = useLibraryInvalidation();
  return useMutation({
    mutationFn: ({
      collectionId,
      documentId,
      member,
    }: {
      collectionId: string;
      documentId: string;
      member: boolean;
    }) =>
      member
        ? api.addToCollection(collectionId, documentId)
        : api.removeFromCollection(collectionId, documentId),
    onSettled: invalidate,
  });
}
